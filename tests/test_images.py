"""Canonical image types and the image model ID.

Two contracts meet here. The first is the canonical representation: an image
request and response round-trip through ``to_dict`` and ``from_dict`` with every
field emitted, and every state the representation cannot hold is refused rather
than dropped. The sharpest of those refusals is the
image payload: a result entry carrying both a base64 payload and a URL, or
neither, is refused at construction and at parse, because a caller reads
whichever one is set and neither of those states defines that reading. The URL
it does carry is untrusted provider output, so these tests also pin that
canonical parsing screens it in no way at all.

The second is the public model ID ``image-default``, which resolves onto the
configured image routes exactly as ``text-default`` and ``embedding-default``
resolve onto theirs, and appears in the model listing only when an image block is
configured — carrying the primary route's declared capabilities, so a caller can
tell a model that generates from one that generates and edits.

Nothing here reaches the network: every endpoint is an IP literal and every
request is built by hand.
"""

import dataclasses
import json

import pytest

from model_gateway.capabilities import Capability
from model_gateway.config.routes import Operation
from model_gateway.domain.generation import Usage
from model_gateway.domain.images import (
    IMAGE_RESPONSE_FORMATS,
    GeneratedImage,
    ImageRequest,
    ImageResponse,
    as_images,
)
from model_gateway.errors import ErrorReason, ModelNotFoundError
from model_gateway.registry import ModelRegistry

PUBLIC_ADDRESS = "93.184.216.34"
FALLBACK_ADDRESS = "93.184.216.35"

PAYLOAD = "aW1hZ2UtYnl0ZXM="

IMAGE = GeneratedImage(b64_json=PAYLOAD)

REQUEST = ImageRequest(
    model="image-default",
    prompt="A futuristic city at sunset",
    size="1024x1024",
    response_format="b64_json",
    n=1,
    metadata={"trace": {"sampled": True}},
)

RESPONSE = ImageResponse(
    model="image-default",
    upstream_model="image-model",
    created=1700000000,
    data=(IMAGE,),
    usage=Usage(total_tokens=1),
    invocation_id="inv-7f3c1a90",
)


def request(**overrides) -> ImageRequest:
    values = {"model": "image-default", "prompt": "A futuristic city at sunset"}
    values.update(overrides)
    return ImageRequest(**values)


def response(**overrides) -> ImageResponse:
    values = {
        "model": "image-default",
        "upstream_model": "image-model",
        "created": 1700000000,
        "data": (IMAGE,),
        "usage": Usage(total_tokens=1),
        "invocation_id": "inv-7f3c1a90",
    }
    values.update(overrides)
    return ImageResponse(**values)


# --------------------------------------------------------------------------- the generated image


def test_a_generated_image_round_trips():
    assert GeneratedImage.from_dict(IMAGE.to_dict()) == IMAGE


def test_a_generated_image_round_trips_as_a_url():
    by_url = GeneratedImage(url="https://images.example/a.png", revised_prompt="a city")

    assert GeneratedImage.from_dict(by_url.to_dict()) == by_url


def test_a_generated_image_emits_every_field():
    assert IMAGE.to_dict() == {"b64_json": PAYLOAD, "url": None, "revised_prompt": None}


def test_a_generated_image_keeps_the_revised_prompt_beside_the_payload():
    revised = GeneratedImage(b64_json=PAYLOAD, revised_prompt="A city at dusk")

    assert revised.to_dict() == {
        "b64_json": PAYLOAD,
        "url": None,
        "revised_prompt": "A city at dusk",
    }
    assert GeneratedImage.from_dict(revised.to_dict()) == revised


def test_a_generated_image_is_frozen_and_hashable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        IMAGE.b64_json = "another-payload"
    assert len({IMAGE, GeneratedImage(b64_json=PAYLOAD)}) == 1


def test_a_generated_image_refuses_carrying_both_payloads():
    """The two are alternatives, so a caller reading one would be choosing."""
    with pytest.raises(ValueError, match="not both"):
        GeneratedImage(b64_json=PAYLOAD, url="https://images.example/a.png")


def test_a_generated_image_refuses_carrying_neither_payload():
    with pytest.raises(ValueError, match="carries neither"):
        GeneratedImage()

    with pytest.raises(ValueError, match="carries neither"):
        GeneratedImage(revised_prompt="A city at dusk")


def test_a_generated_image_refuses_both_payloads_through_from_dict():
    with pytest.raises(ValueError, match="not both"):
        GeneratedImage.from_dict({"b64_json": PAYLOAD, "url": "https://images.example/a.png"})


def test_a_generated_image_refuses_neither_payload_through_from_dict():
    with pytest.raises(ValueError, match="carries neither"):
        GeneratedImage.from_dict({"b64_json": None, "url": None})


def test_a_generated_image_refuses_an_empty_payload():
    with pytest.raises(ValueError, match="'b64_json' must not be empty"):
        GeneratedImage(b64_json="")

    with pytest.raises(ValueError, match="'url' must not be empty"):
        GeneratedImage(url="")


def test_a_generated_image_refuses_a_key_it_does_not_define():
    with pytest.raises(ValueError, match="unsupported key 'image_url'"):
        GeneratedImage.from_dict({"url": "https://images.example/a.png", "image_url": "x"})


def test_a_generated_image_refuses_a_payload_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        GeneratedImage.from_dict([PAYLOAD])


def test_a_generated_image_url_is_carried_through_unscreened():
    """A generated image's url is provider output, not a URL the gateway vouches for.

    The type accepts any non-empty string, so surviving canonical parsing says
    nothing about where the URL points: the gateway does not resolve it, apply
    the URL policy to it, pin a connection to it or fetch it, here or anywhere
    else. A caller that fetches it does so on its own authority, which is why a
    private address is representable and why this value must not be fed back to
    the gateway as a URL it may fetch — the mirror image of the remote image
    input the HTTP layer refuses.
    """
    url = "http://127.0.0.1:9000/internal.png"

    image = GeneratedImage(url=url)

    assert image.url == url
    assert GeneratedImage.from_dict(image.to_dict()) == image


# --------------------------------------------------------------------------- the request


def test_a_request_round_trips_with_every_optional_field_unset():
    minimal = request()

    assert minimal.to_dict() == {
        "model": "image-default",
        "prompt": "A futuristic city at sunset",
        "size": None,
        "response_format": None,
        "n": None,
        "metadata": None,
        "surface": None,
    }
    assert ImageRequest.from_dict(minimal.to_dict()) == minimal


def test_a_request_round_trips_with_every_optional_field_set():
    document = REQUEST.to_dict()

    assert document == {
        "model": "image-default",
        "prompt": "A futuristic city at sunset",
        "size": "1024x1024",
        "response_format": "b64_json",
        "n": 1,
        "metadata": {"trace": {"sampled": True}},
        "surface": None,
    }
    assert ImageRequest.from_dict(document) == REQUEST


def test_a_request_round_trips_a_stamped_surface():
    stamped = request(surface="openai_compat")

    assert ImageRequest.from_dict(stamped.to_dict()) == stamped
    assert ImageRequest.from_dict(stamped.to_dict()).surface == "openai_compat"


def test_a_request_round_trip_is_stable_under_reparsing():
    once = ImageRequest.from_dict(REQUEST.to_dict())
    twice = ImageRequest.from_dict(once.to_dict())

    assert twice == once


def test_a_request_keeps_an_unstated_count_distinct_from_one():
    unstated = request(n=None)

    assert unstated.n is None
    assert unstated != request(n=1)
    assert ImageRequest.from_dict(unstated.to_dict()).n is None


def test_a_request_is_frozen_and_hashable():
    assert len({request(), request()}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        REQUEST.model = "text-default"


def test_a_request_refuses_an_empty_model_id():
    with pytest.raises(ValueError, match="model must not be empty"):
        request(model="")


def test_a_request_refuses_an_empty_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        request(prompt="")


def test_a_request_keeps_its_prompt_verbatim():
    assert request(prompt="  A city  ").prompt == "  A city  "


@pytest.mark.parametrize("size", ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"])
def test_a_request_accepts_the_documented_size_form(size):
    assert request(size=size).size == size


@pytest.mark.parametrize(
    "size",
    ["1024", "1024x", "x1024", "1024X1024", "1024x1024x1024", "large", "0x0", "1024x0", "0x1024"],
)
def test_a_request_refuses_a_size_outside_the_documented_form(size):
    with pytest.raises(ValueError, match="must be the documented 'WxH' form"):
        request(size=size)


def test_a_request_refuses_a_padded_size():
    with pytest.raises(ValueError, match="must be the documented 'WxH' form"):
        request(size=" 1024x1024 ")


def test_a_request_refuses_an_empty_size():
    with pytest.raises(ValueError, match="size must not be empty"):
        request(size="")


def test_the_documented_response_formats_are_the_two_wire_representations():
    assert frozenset({"url", "b64_json"}) == IMAGE_RESPONSE_FORMATS


@pytest.mark.parametrize("response_format", sorted(IMAGE_RESPONSE_FORMATS))
def test_a_request_accepts_a_documented_response_format(response_format):
    assert request(response_format=response_format).response_format == response_format


def test_a_request_refuses_an_undocumented_response_format():
    with pytest.raises(ValueError, match="must be one of 'b64_json', 'url'"):
        request(response_format="hex")


def test_a_request_refuses_a_non_positive_count():
    with pytest.raises(ValueError, match="n must be at least 1"):
        request(n=0)

    with pytest.raises(ValueError, match="n must be at least 1"):
        request(n=-1)


def test_a_request_refuses_a_non_integer_count():
    with pytest.raises(ValueError, match="n must be an integer"):
        request(n="1")

    with pytest.raises(ValueError, match="n must be an integer"):
        request(n=True)


def test_a_request_keeps_absent_metadata_distinct_from_empty_metadata():
    absent = request(metadata=None)
    empty = request(metadata={})

    assert absent.metadata is None
    assert empty.metadata == {}
    assert ImageRequest.from_dict(absent.to_dict()).metadata is None
    assert ImageRequest.from_dict(empty.to_dict()).metadata == {}


def test_a_request_holds_its_metadata_immutably():
    metadata = request(metadata={"trace": {"sampled": True}}).metadata

    with pytest.raises(TypeError):
        metadata["trace"] = "no"


def test_a_request_refuses_metadata_that_is_not_json():
    with pytest.raises(ValueError, match="cannot carry"):
        request(metadata={"bad": {1, 2}})


def test_a_request_refuses_metadata_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        request(metadata=["trace"])


def test_a_request_refuses_a_key_it_does_not_define():
    with pytest.raises(ValueError, match="unsupported key 'quality'"):
        ImageRequest.from_dict({"model": "image-default", "prompt": "A city", "quality": "hd"})


# --------------------------------------------------------------------------- the response


def test_a_response_round_trips():
    assert ImageResponse.from_dict(RESPONSE.to_dict()) == RESPONSE


def test_a_response_round_trip_is_stable_under_reparsing():
    once = ImageResponse.from_dict(RESPONSE.to_dict())
    twice = ImageResponse.from_dict(once.to_dict())

    assert twice == once


def test_a_response_emits_every_field():
    assert RESPONSE.to_dict() == {
        "model": "image-default",
        "upstream_model": "image-model",
        "created": 1700000000,
        "data": [{"b64_json": PAYLOAD, "url": None, "revised_prompt": None}],
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": 1,
            "cached_tokens": None,
            "estimated_fields": [],
        },
        "invocation_id": "inv-7f3c1a90",
    }


def test_a_response_preserves_the_order_of_the_images():
    second = GeneratedImage(url="https://images.example/b.png")
    parsed = ImageResponse.from_dict(response(data=(IMAGE, second)).to_dict())

    assert [image.b64_json for image in parsed.data] == [PAYLOAD, None]
    assert parsed.data[1].url == "https://images.example/b.png"


def test_a_response_is_frozen_and_hashable():
    assert len({response(), response()}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        RESPONSE.created = 1


def test_a_response_refuses_an_empty_image_list():
    with pytest.raises(ValueError, match="at least one generated image"):
        response(data=())


def test_a_response_refuses_an_entry_that_is_not_a_generated_image():
    with pytest.raises(ValueError, match=r"data\[0\] must be a GeneratedImage"):
        response(data=({"b64_json": PAYLOAD},))


def test_a_response_refuses_a_negative_creation_time():
    with pytest.raises(ValueError, match="created must be at least 0"):
        response(created=-1)


def test_a_response_refuses_a_non_integer_creation_time():
    with pytest.raises(ValueError, match="created must be an integer"):
        response(created="1700000000")


def test_a_response_refuses_a_missing_creation_time():
    payload = {key: value for key, value in RESPONSE.to_dict().items() if key != "created"}

    with pytest.raises(ValueError, match="'created' must be an integer, got null"):
        ImageResponse.from_dict(payload)


def test_a_response_refuses_usage_that_is_not_a_usage_record():
    with pytest.raises(ValueError, match="usage must be a Usage"):
        response(usage={"total_tokens": 1})


def test_a_response_refuses_a_missing_usage():
    payload = {key: value for key, value in RESPONSE.to_dict().items() if key != "usage"}

    with pytest.raises(ValueError, match="usage must be a JSON object, got null"):
        ImageResponse.from_dict(payload)


def test_a_response_refuses_an_empty_invocation_id():
    with pytest.raises(ValueError, match="invocation_id must not be empty"):
        response(invocation_id="")


def test_a_response_refuses_an_empty_upstream_model():
    with pytest.raises(ValueError, match="upstream_model must not be empty"):
        response(upstream_model="")


def test_a_response_refuses_an_empty_model():
    with pytest.raises(ValueError, match="model must not be empty"):
        response(model="")


def test_a_response_refuses_a_key_it_does_not_define():
    payload = {**RESPONSE.to_dict(), "object": "list"}

    with pytest.raises(ValueError, match="unsupported key 'object'"):
        ImageResponse.from_dict(payload)


def test_a_response_refuses_a_payload_that_is_not_an_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        ImageResponse.from_dict([])


def test_as_images_coerces_a_sequence_of_generated_images():
    assert as_images("data", [IMAGE]) == (IMAGE,)
    assert as_images("data", ()) == ()


def test_as_images_refuses_an_entry_that_is_not_a_generated_image():
    with pytest.raises(ValueError, match=r"data\[1\] must be a GeneratedImage"):
        as_images("data", [IMAGE, "https://images.example/a.png"])


# --------------------------------------------------------------------------- the registry


def text_block(**overrides: str) -> dict[str, str]:
    env = {
        "TEXT_LLM_ENDPOINT": f"https://{PUBLIC_ADDRESS}/v1",
        "TEXT_LLM_MODEL_ID": "primary-model",
        "TEXT_LLM_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def image_block(**overrides: str) -> dict[str, str]:
    env = {
        "IMAGE_GEN_ENDPOINT": f"https://{PUBLIC_ADDRESS}/images/v1",
        "IMAGE_GEN_MODEL_ID": "image-model",
        "IMAGE_GEN_FORMAT": "openai",
    }
    env.update(overrides)
    return env


def image_fallback(model_id: str, **entry: object) -> dict[str, str]:
    route = {
        "endpoint": f"https://{FALLBACK_ADDRESS}/images/v1",
        "model_id": model_id,
        "format": "openai",
    }
    route.update(entry)
    return {"IMAGE_GEN_ROUTES_JSON": json.dumps([route])}


def registry(env: dict[str, str]) -> ModelRegistry:
    return ModelRegistry.from_env(env)


def test_image_default_resolves_to_the_configured_image_routes():
    models = registry({**text_block(), **image_block()})

    routes = models.routes_for("image-default")

    assert [route.operation for route in routes] == [Operation.IMAGE]
    assert [route.model_id for route in routes] == ["image-model"]


def test_image_default_resolves_to_its_fallbacks_in_configured_order():
    models = registry(
        {**text_block(), **image_block(), **image_fallback("image-fallback")}
    )

    routes = models.routes_for("image-default")

    assert [(route.index, route.model_id) for route in routes] == [
        (0, "image-model"),
        (1, "image-fallback"),
    ]


def test_image_default_is_not_served_by_another_operations_routes():
    models = registry({**text_block(), **image_block()})

    assert [route.operation for route in models.routes_for("text-default")] == [Operation.TEXT]
    assert [route.operation for route in models.routes_for("image-default")] == [Operation.IMAGE]


def test_image_default_advertises_the_capability_its_primary_route_declares():
    models = registry(image_block())

    assert models.get("image-default").capabilities == frozenset({Capability.GENERATION})


def test_image_default_advertises_editing_only_when_it_is_declared():
    """A caller reads a model's contract off the listing."""
    declared = registry(image_block(IMAGE_GEN_CAPABILITIES="generation,editing"))
    undeclared = registry(image_block(IMAGE_GEN_CAPABILITIES="generation"))

    assert declared.get("image-default").capabilities == frozenset(
        {Capability.GENERATION, Capability.EDITING}
    )
    assert Capability.EDITING not in undeclared.get("image-default").capabilities


def test_a_fallback_routes_capabilities_are_not_advertised_for_the_image_model():
    models = registry(
        {
            **image_block(IMAGE_GEN_CAPABILITIES="generation"),
            **image_fallback("image-fallback", capabilities=["generation", "editing"]),
        }
    )

    assert models.get("image-default").capabilities == frozenset({Capability.GENERATION})


def test_image_default_advertises_no_text_or_embedding_limit():
    models = registry(image_block())

    info = models.get("image-default")

    assert (info.context_window, info.max_output_tokens, info.dimensions) == (None, None, None)


def test_an_unknown_public_model_id_raises_the_model_not_found_error():
    models = registry(image_block())

    with pytest.raises(ModelNotFoundError) as error:
        models.get("image-default-2")

    assert error.value.reason is ErrorReason.MODEL_NOT_FOUND
    assert error.value.http_status == 404


def test_an_upstream_image_model_name_is_not_a_public_model_id():
    models = registry(image_block())

    with pytest.raises(ModelNotFoundError):
        models.get("image-model")

    with pytest.raises(ModelNotFoundError):
        models.routes_for("openai/image-model")


def test_a_public_model_id_whose_operation_is_unconfigured_is_not_found():
    models = registry(text_block())

    with pytest.raises(ModelNotFoundError):
        models.get("image-default")

    with pytest.raises(ModelNotFoundError):
        models.routes_for("image-default")


def test_list_models_omits_image_generation_when_it_is_unconfigured():
    models = registry({**text_block(), **image_block()})
    without = registry(text_block())

    assert [info.id for info in models.list_models()] == ["text-default", "image-default"]
    assert [info.id for info in without.list_models()] == ["text-default"]
    assert "image-default" not in {info.id for info in without.list_models()}


def test_list_models_omits_image_generation_while_keeping_the_other_operations():
    models = registry(
        {
            **text_block(),
            "EMBEDDINGS_ENDPOINT": f"https://{PUBLIC_ADDRESS}/embeddings/v1",
            "EMBEDDINGS_MODEL_ID": "embedding-model",
            "EMBEDDINGS_FORMAT": "openai",
        }
    )

    assert [info.id for info in models.list_models()] == [
        "text-default",
        "embedding-default",
    ]


def test_the_image_entry_is_the_one_the_listing_and_the_lookup_agree_on():
    models = registry({**text_block(), **image_block()})

    listed = {info.id: info for info in models.list_models()}

    assert models.get("image-default") == listed["image-default"]
    assert listed["image-default"].operation is Operation.IMAGE
