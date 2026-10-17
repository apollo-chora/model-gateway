"""The Invoke RPC's image modality, driven the way the renderer drives it.

The contract carries one ``Invoke`` RPC and a ``response_modality`` field, so an
image request is an ``Invoke`` that declares ``IMAGE``. These tests pin the two
halves of that: the canonical request the handler builds from the message parts,
and the answer it returns — an output item carrying the image part, which is the
only thing the Go client accepts as an image.
"""

import grpc
import pytest

from model_gateway.config import Settings
from model_gateway.config.routes import Operation, Route
from model_gateway.domain import ImageResponse, Usage
from model_gateway.domain.images import GeneratedImage
from model_gateway.errors import InvalidArgumentError
from model_gateway.grpc import gateway_pb2 as pb
from model_gateway.grpc import gateway_pb2_grpc, mapping
from model_gateway.grpc.identity import Authenticator
from model_gateway.grpc.invoke import Invoke
from model_gateway.security.api_keys import ApiKeySet

API_KEY = "sk-gateway-test-key"
TENANT = "tenant-a"
PUBLIC_MODEL = "image-default"
UPSTREAM_MODEL = "upstream-image"

_HOST = "127.0.0.1"


def image_route() -> Route:
    return Route(
        operation=Operation.IMAGE,
        index=0,
        endpoint="https://upstream.internal.example/v1",
        base_url="https://upstream.internal.example/v1",
        model_id=UPSTREAM_MODEL,
        format="openai",
        api_key="sk-upstream",
        capabilities=frozenset(),
        extra_headers={},
        allow_absolute_path_url=False,
    )


def image_request(**overrides) -> pb.InvokeRequest:
    fields = {
        "model": PUBLIC_MODEL,
        "messages": (
            pb.Message(
                role=pb.ROLE_USER,
                content=(pb.ContentPart(text=pb.TextPart(text="a fox on the moon")),),
            ),
        ),
        "response_modality": pb.RESPONSE_MODALITY_IMAGE,
        "surface": "qgen",
    }
    fields.update(overrides)
    return pb.InvokeRequest(**fields)


def identity():
    settings = Settings.from_env({"GATEWAY_AUTH_ENABLED": "false"})
    authenticator = Authenticator(
        ApiKeySet.from_settings(settings),
        auth_enabled=settings.auth_enabled,
        allow_anonymous_health=settings.allow_anonymous_health,
    )
    return authenticator


class _RecordingImageService:
    """A stand-in for the image operation's entry point, recording what it was handed."""

    def __init__(self, response: ImageResponse) -> None:
        self._response = response
        self.calls: list[object] = []

    async def generate(self, request, *, namespace=None, tenant=None):  # noqa: ANN001, ARG002
        self.calls.append(request)
        return self._response

    async def invoke(self, request, *, namespace=None, tenant=None):  # noqa: ANN001, ARG002
        raise AssertionError("the text operation must not serve an image request")


def an_image_response() -> ImageResponse:
    return ImageResponse(
        model=PUBLIC_MODEL,
        upstream_model=UPSTREAM_MODEL,
        created=1792232500,
        data=(GeneratedImage(b64_json="UklGRg=="),),
        usage=Usage(),
        invocation_id="inv-image-1",
    )


class TestTheImageRequestMapping:
    def test_the_prompt_is_the_single_text_part(self) -> None:
        request = mapping.image_request_from_proto(
            image_request(), identity=identity().identity(_FakeContext())
        )

        assert request.prompt == "a fox on the moon"
        assert request.model == PUBLIC_MODEL
        assert request.surface == "qgen"

    def test_two_messages_are_refused_rather_than_joined(self) -> None:
        with pytest.raises(InvalidArgumentError, match="exactly one message"):
            mapping.image_request_from_proto(
                image_request(
                    messages=(
                        pb.Message(
                            role=pb.ROLE_USER,
                            content=(pb.ContentPart(text=pb.TextPart(text="a")),),
                        ),
                        pb.Message(
                            role=pb.ROLE_USER,
                            content=(pb.ContentPart(text=pb.TextPart(text="b")),),
                        ),
                    )
                ),
                identity=identity().identity(_FakeContext()),
            )

    def test_two_text_parts_are_refused_rather_than_joined(self) -> None:
        with pytest.raises(InvalidArgumentError, match="exactly one text part"):
            mapping.image_request_from_proto(
                image_request(
                    messages=(
                        pb.Message(
                            role=pb.ROLE_USER,
                            content=(
                                pb.ContentPart(text=pb.TextPart(text="a")),
                                pb.ContentPart(text=pb.TextPart(text="b")),
                            ),
                        ),
                    )
                ),
                identity=identity().identity(_FakeContext()),
            )

    def test_an_image_part_is_refused_rather_than_dropped(self) -> None:
        with pytest.raises(InvalidArgumentError, match="cannot carry a image part"):
            mapping.image_request_from_proto(
                image_request(
                    messages=(
                        pb.Message(
                            role=pb.ROLE_USER,
                            content=(
                                pb.ContentPart(text=pb.TextPart(text="redraw this")),
                                pb.ContentPart(
                                    image=pb.ImagePart(data="AAAA", media_type="image/png")
                                ),
                            ),
                        ),
                    )
                ),
                identity=identity().identity(_FakeContext()),
            )


class TestTheImageAnswerMapping:
    def test_bytes_become_an_image_output_item(self) -> None:
        answer = mapping.image_response_to_proto(an_image_response())

        assert answer.finish_reason == pb.FINISH_REASON_STOP
        assert len(answer.output) == 1
        part = answer.output[0].content
        assert part.WhichOneof("part") == "image"
        assert part.image.data == "UklGRg=="
        assert part.image.WhichOneof("source") == "data"

    def test_a_url_becomes_the_image_location(self) -> None:
        response = ImageResponse(
            model=PUBLIC_MODEL,
            upstream_model=UPSTREAM_MODEL,
            created=1792232500,
            data=(GeneratedImage(url="https://example.com/a.png"),),
            usage=Usage(),
            invocation_id="inv-image-2",
        )

        answer = mapping.image_response_to_proto(response)

        assert answer.output[0].content.image.url == "https://example.com/a.png"


class _FakeContext:
    """The authenticator reads the call's metadata; auth is disabled for mapping tests."""

    def invocation_metadata(self):
        return ()


class _ImageOnlyServicer(gateway_pb2_grpc.ModelGatewayServiceServicer):
    def __init__(self, invoke: Invoke) -> None:
        self._invoke = invoke

    async def Invoke(self, request, context):
        return await self._invoke.invoke(request, context)


class TestTheInvokeDispatch:
    async def test_an_image_modality_is_served_by_the_image_operation(self) -> None:
        service = _RecordingImageService(an_image_response())
        settings = Settings.from_env({"GATEWAY_AUTH_ENABLED": "false"})
        authenticator = Authenticator(
            ApiKeySet.from_settings(settings),
            auth_enabled=settings.auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
        server = grpc.aio.server()
        port = server.add_insecure_port(f"{_HOST}:0")
        gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
            _ImageOnlyServicer(Invoke(authenticator, service)), server
        )
        await server.start()
        channel = grpc.aio.insecure_channel(f"{_HOST}:{port}")
        try:
            stub = gateway_pb2_grpc.ModelGatewayServiceStub(channel)
            answer = await stub.Invoke(image_request())
        finally:
            await channel.close()
            await server.stop(grace=None)

        assert len(service.calls) == 1
        assert answer.output[0].content.WhichOneof("part") == "image"

    async def test_a_multimodal_answer_is_refused(self) -> None:
        service = _RecordingImageService(an_image_response())
        settings = Settings.from_env({"GATEWAY_AUTH_ENABLED": "false"})
        authenticator = Authenticator(
            ApiKeySet.from_settings(settings),
            auth_enabled=settings.auth_enabled,
            allow_anonymous_health=settings.allow_anonymous_health,
        )
        server = grpc.aio.server()
        port = server.add_insecure_port(f"{_HOST}:0")
        gateway_pb2_grpc.add_ModelGatewayServiceServicer_to_server(
            _ImageOnlyServicer(Invoke(authenticator, service)), server
        )
        await server.start()
        channel = grpc.aio.insecure_channel(f"{_HOST}:{port}")
        try:
            stub = gateway_pb2_grpc.ModelGatewayServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.Invoke(image_request(response_modality=pb.RESPONSE_MODALITY_MULTIMODAL))
        finally:
            await channel.close()
            await server.stop(grace=None)

        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert service.calls == []
