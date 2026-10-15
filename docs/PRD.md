# Product Requirements Document
# Vendor-Neutral AI Model Gateway

**Version:** 1.30  
**Status:** Implementation-ready specification  
**Implementation language:** Python  
**Architecture:** Standalone, self-hostable, multi-provider AI gateway  
**Protocols:** gRPC, HTTP/JSON  
**Configuration:** Environment variables  
**Primary deployment:** Containerized service  
**Supported capabilities:** Text generation, chat, tool calling, vision, structured output, image generation, embeddings

**Revision history:**

- **1.27** — Records idempotency on the gRPC contract and the mechanism that answers a repeat (§5.1, §5.2, §12), and qualifies what two HTTP payloads say about the gateway's own accounting now that they carry two of its fields. (§5.1) `InvokeRequest` carries `idempotency_key` as field 15, and the field is `optional string`: an absent key and an empty one stay distinct across the boundary, exactly as the canonical record keeps them distinct, so the mapping reads it through `HasField` and a stated empty key is refused as an invalid argument rather than read as no key at all. (§5.2) `InvokeResponse` carries `idempotency_status` as field 13, a closed enum whose zero value is the unspecified member rather than a status, so no answer leaves it unset; the four values are stated, and `replayed` is the one a caller reads to learn that the provider was not reached again. (§12) The mechanism is recorded as implemented: the fingerprint covers the payload and not the key, the gateway-minted invocation identifier is excluded from it, only a completed answer is recorded, the store is in memory and therefore single-instance, and a repeat under the same key reaches the caller without reaching the provider. **Concurrent identical requests are coalesced, not just sequential repeats.** The first caller claims the key and every later caller with the same fingerprint waits on that claim's outcome instead of starting a second invocation. Without this, two simultaneous retries — the situation idempotency exists to handle — both call the provider. Cancellation belongs to the caller who owns it: a waiter going away does not cancel the invocation it is waiting on, and an owner failing gives every waiter that failure rather than starting another provider attempt inside the same request, before releasing the claim so a later caller may retry.

The record is consulted after the model is resolved and before any provider is called, so an unknown model is a model-not-found error even when the key has a cached record. (§4.4, §4.5) The chat completions and legacy completions payloads accept `idempotency_key` on the request and answer `idempotency_status`, The Responses payload carries both, like the other two text endpoints, because this gateway extends the Responses object with its own fields rather than treating it as closed. The statements that said the gateway's own accounting is deliberately absent from those payloads are qualified rather than left to contradict the implementation. The images and embeddings payloads carry neither field either, so their statements stand unchanged.

- **1.30** — Records usage and cost accounting as implemented (§13), and corrects it against a review of what it actually does. Every invocation that produces an answer leaves a record, for all three operations and not only for text: the record is built in the inference service, which is the only layer that knows which route actually answered, so a fallback answer is attributed to the route that served it rather than to the plan's primary — a record credited to the primary would name a route that produced nothing and advertise capabilities the answering route may not have. The record names the invocation identifier and timestamp, the serving route and its advertised capabilities, the public model, the upstream model, the operation, the token counts and their provenance, the provider latency, the attempt count, whether a fallback was used, and the estimated cost. The attempt count and the fallback flag reach the record from the walk rather than from the answer, because the canonical embeddings and image answers have no field for either (§5.2) while the record has one for both. Token provenance travels with each count, because rendering an unreported count as zero turns an unknown quantity into an asserted one; the wire layers make that trade-off and state it, the record does not. A count the provider never reported is not priced either, because the estimate is of *known* usage. Cost comes only from configured pricing, per million tokens, keyed by public model ID; a deployment that configures none records usage and no cost, and a malformed pricing value is refused at startup rather than partially applied. A tenant is recorded only where the request stated one, so an embeddings or image record carries none: the canonical requests for those operations have no tenant field, and the tenant an invocation acts under is the credential's (§5.1). Storage is in memory and bounded, the oldest record dropped once the store is full, so the replay guarantee is weakened in the way the bound states; the durable modes are named but not implemented. Recording is neither part of serving the request nor part of idempotency's own record: a sink that fails is logged rather than allowed to fail the invocation, because the answer has already been produced and the caller is waiting on it, and a coalesced waiter is not recorded at all because it did not reach the provider.
- **1.29** — Records the two corrections the second review of the idempotency mechanism required, and a consequence it surfaced. (§12) A coalesced waiter now reports ``replayed`` rather than ``new``, because it did not initiate provider execution and the status names this request's handling rather than the invocation's — so a coalesced caller is told the truth about whether its request reached the provider. The shared anonymous namespace that results from disabling authentication is stated as the deliberate consequence it is, with the reason that inventing isolation there would break the identity rules.
- **1.28** — Corrects the idempotency mechanism against a review of what it does, and records the corrections. (§12) Three semantics were wrong. A repeat carrying a different fingerprint was executed as a new request that overwrote the stored record, letting one key oscillate between unrelated payloads and hiding a caller's reuse of a key it believed unique; it is now refused with `idempotency_conflict` (409 Conflict / gRPC `ALREADY_EXISTS`) before any provider is called, leaving the record intact. The store was keyed by the caller's key alone, so two tenants using the same key overwrote each other's records — a cross-tenant interference channel; the key is now namespaced by the trusted principal derived from the credential, its identifier and its tenant. And simultaneous identical requests both missed the store and both called the provider, so the mechanism did not handle the situation it exists for; the first caller now claims the key and later callers with the same fingerprint wait on that claim's outcome, with cancellation belonging to the caller who owns it. The fingerprint was also taken over `repr()`, which is an implementation representation that changes when the class does; it is now taken over a canonical JSON encoding. (§5.4) The error vocabulary gains the conflict category, and gRPC's `ALREADY_EXISTS` the status name, for it.
- **1.26** — Records the gRPC service as implemented (§5, §18) and tightens the contract's canonical fidelity. (§5) `ModelGatewayService` is served on `CHORA_GRPC_PORT` beside the HTTP listener, started and stopped with the application, over four RPCs — `Invoke`, `InvokeStream`, `Embed` and `ListModels` — with the standard gRPC Health Checking Protocol on the same port. No policy lives in the gRPC layer: each RPC authenticates the call, reconciles the claimed tenant against the credential, maps the protobuf request onto the canonical domain and calls the inference service, so both transports reach identical behaviour. The canonical-fidelity rule is drawn as one line: a field the canonical model types is typed protobuf, and a field it holds as arbitrary JSON is canonical-domain JSON bytes, because the only JSON-ish well-known protobuf types represent every number as a double and so change both the type and the value of a JSON integer. Metadata on both requests, a tool's parameters, a JSON Schema and tool-call arguments are therefore JSON text. `tool_choice` is in the contract and §5.1's table is corrected to list it. (§18) The health protocol is served without authentication and with the package's own implementation, and it is set to `SERVING` when the listener starts and enters graceful shutdown before the listener stops, so a watcher is told the transition rather than left with a vanished connection. (§16) The canonical `StreamEvent` is corrected to accept an empty `raw_finish_reason`, matching `InvocationResponse`; the two had diverged, and the divergence meant the streaming path could not preserve the absent-versus-empty distinction the gateway's own transport requires.
- **1.24** — Corrects two claims the Responses and legacy completion sections make about what their wire shapes can express, records the request validation the three inference endpoints now apply, and records the capability-list changes a review produced. (§4.5) The ordering claim is corrected: the Responses renderer emits every content part of the assistant turn first and then every tool call, so the representable grammar is `(ContentPart)* (ToolCall)*` and an arbitrary interleaving of content and tool calls is not representable; the previous wording claimed the order of the canonical output survives the translation wholesale. The `interleaved_output` refusal reason is reworded: the refusal was stated as a limit of the wire format, and it is not — the wire could carry the trailing content as a second `message` item. The real reason is that the canonical model records one assistant turn, so emitting two message items would invent a turn boundary the caller never sent, and the wording now says so, so the refusal is not 'fixed' by emitting two items. (§4.4, §4.5) Both new endpoints, and the chat endpoint beside them, now validate the wire type of a recognised parameter before testing whether its value is the one the gateway already behaves like: a recognised parameter whose JSON type is wrong — `{"echo": 0}`, `{"echo": ""}`, `{"echo": []}`, `{"echo": {}}`, or `{"n": true}` and `{"best_of": true}` where `1` is meant, because Python's `True == 1` made a boolean a silently accepted stand-in for an integer — is refused as an invalid request (400) naming the parameter and the JSON type actually supplied, rather than being mistaken for 'nothing requested' and skipping the capability refusal a request the gateway cannot serve would otherwise raise; only a correctly typed value that is not the upstream default raises the capability error. The usage-rendering rule is recorded for both endpoints: a token count the provider never reported is rendered as `0`, because the wire usage shape cannot distinguish 'zero' from 'unknown'. The capability-list changes are recorded in the same version: on the Responses endpoint `top_p` and `tool_choice` were listed as not carried and are now mapped, onto the canonical generation config and the canonical tool choice respectively, exactly as the chat endpoint maps them; Responses `metadata` is now mapped onto the request metadata the invocation record already holds (§5.2) rather than accepted only when falsy; and `input_image` is accepted as a content part, becoming a canonical image part under the same `vision` capability rule the chat endpoint's image part is held to. The identical default-equivalence predicates and the same hole existed in the chat endpoint's request validation, which received the same wire-type rule (§4.4).
- **1.23** — Records the Responses and legacy completion endpoints as implemented (§4.4, §4.5). (§4.5) `POST /v1/responses` is served by the gateway core through the ordinary text-generation operation, so a Responses-compatible upstream is not required: the canonical invocation is the one `POST /v1/chat/completions` builds and the canonical answer is the one the chat binding produces, and rendering that answer into the Responses representation is the gateway's own work. The accepted request is the documented subset — `model`, `input`, `instructions`, `tools`, `temperature`, `max_output_tokens`, `stream` and `response_format` — and a key the gateway does not understand is refused with an invalid-request error naming it rather than dropped. A key the Responses format defines and this endpoint does not carry is refused with a capability error naming the feature it would request — `top_p`, `tool_choice`, `previous_response_id`, `text`, `truncation`, `reasoning`, `conversation`, `prompt` and the prompt-cache parameters, `max_tool_calls`, `moderation`, `service_tier`, `user`, `top_logprobs` and `parallel_tool_calls` — while a parameter whose value is the one the gateway already behaves like is accepted: `stream: false`, `store: false` and the falsy defaults `background: false`, `include: false`, `metadata: false` and `stream_options: false`. The ordered canonical answer is rendered into the ordered `output` array, which is what the Responses representation can express and the chat shape cannot: one assistant turn's content parts become one `message` item's content list in the order the model emitted them, each tool call becomes a `function_call` item after it, and a tool-call-only answer carries no `message` item. Two canonical answers the representation cannot express are refused rather than approximated — an answer carrying an image, naming `image_output`, because a `message` item's content holds output text and refusals and nothing else, and an answer whose content continues after a tool call, naming `interleaved_output`, because one `output` array is one assistant turn and a turn that calls a function ends there. The response is the documented subset of the Responses object the endpoint states truthfully — `id`, `object: "response"`, `created_at`, `model`, `status`, the ordered `output` and a `usage` object — and the gateway's own accounting is deliberately absent because the shape has no field for it. **Streaming is refused explicitly rather than faked.** The Responses streaming protocol is an item-lifecycle protocol whose events carry item ids, output and content indices and whole-item boundaries, and the canonical stream of §16 carries none of them, so `stream: true` is refused with a capability error naming `streaming` before the model is resolved and before any provider is called, rather than answered with a complete, non-streamed body; the endpoint serves the non-streaming Responses representation only, a documented subset limitation. (§4.4) `POST /v1/completions` is served too, through the same text-generation operation: the prompt becomes a single canonical user message and the sampling parameters map onto the canonical generation config, and the response is the documented legacy shape — `id` (`cmpl-…`), `object: "text_completion"`, `created`, `model`, one choice carrying `text`, `index` and `finish_reason`, and `usage`. A recognised parameter the legacy shape cannot honour is refused with a capability error naming the feature — `n` above one, `logprobs`, `echo`, `best_of` above one, and any stated `suffix` — while the upstream default that already matches gateway behaviour is accepted (`n: 1`, `best_of: 1`, `logprobs: false`, `echo: false`), and an array prompt is refused naming `multiple_prompts` rather than narrowed to its first element. An answer carrying a tool call, a refusal or an image is refused naming the feature rather than returned without it. Its streaming is served as Server-Sent Events in the legacy chunk shape, with the terminal contract of §16: `[DONE]` follows a `finish` event and nothing else, and a tool call ends the stream with a terminal `error` event naming `tool_calls` and no `[DONE]`, or is refused as an ordinary error response when it arrives before the first event.
- **1.22** — Records the image entry point as implemented and corrects §6.2 against it, in the same way §6.3 was corrected for embeddings. (§4.6) `POST /v1/images/generations` is served through the image operation of §9.6: the accepted request is the documented subset — `model`, `prompt`, `size`, `response_format`, `n` and `metadata` — and a key the gateway does not understand is refused with an invalid-request error naming it rather than dropped. The response is the documented shape — `created` and one `data` entry per image, each carrying exactly one of `b64_json` or `url` beside `revised_prompt` when the upstream reported one — and the gateway's own accounting (invocation identifier, upstream model, attempt count) is deliberately not in it, because the OpenAI image shape has no field for it. The request refuses what it cannot honour: an empty `prompt`, a `size` outside the documented `WxH` form, a `response_format` outside the two documented wire representations, a non-positive `n`. The answer is checked against the request: a stated `n` is answered with exactly that many images, and an upstream that returned another number is refused as a malformed upstream response — refused on the route that answered wrongly, eligible for another — while an unstated count is checked against none. `POST /v1/images/edits` is served too: the caller's image is accepted as a base64 payload, a base64 data URL, or an array of those, and validated where it arrives, but the invocation itself is refused with a capability error naming `editing` — for a route that does not declare the capability, and equally for one that does, because the canonical image request carries no field for an image input and an edit answered without it would report an image edited that never left the gateway. (§6.2) The stale "not yet implemented" paragraph is replaced: `litellm.aimage_generation` is served, and an image dialect is runnable exactly when its image path is verified to honour the injected client — `openai`, `gemini` and `vertex_ai` — because the image entry point is a different library entry point and each dialect's image path was verified on its own. The dialect refusal runs before any client is built, on the same shared seam as the text and embeddings operations, so an unverified dialect is refused at startup and again at the attempt. The provider names this block was built around that remain outside the set — `azure`, `anthropic`, `openrouter` — are refused rather than run, each for a reason its image path's verification established. (§9.7) The image entry point's verification is recorded as it was measured: which dialects are runnable and why the others are refused, that the transport controls hold on this entry point — a 302 surfaces as an error rather than a hop, and only because the injected client refuses it — and the parameter rule an operator will hit: the library raises `UnsupportedParamsError` for a parameter its provider configuration does not list, before any request, so the binding asks the configuration the library will use and forwards only what it accepts, refusing a stated `n`, `size` or `url` the route's dialect and model cannot honour with a capability error naming it.
- **1.21** — Records the embeddings entry point as implemented and corrects §6.3 against it, states the embedding-space requirement a fallback between embedding routes must meet, and settles the classification of a structurally malformed upstream embeddings answer. (§6.3) The stale "not yet implemented" paragraph is replaced: `litellm.aembedding` is served (§9.6), and an embedding dialect is runnable exactly when its HTTP path is in the verified set of §14.1 — `openai`, `anthropic` and `openrouter` — because the embedding call reaches the wire through the same per-attempt client seam as the text call, so the transport verification is not a separate question the embeddings path had to answer again. The dialect refusal runs before any client is built, on that shared seam, so an unverified dialect is refused at startup — where every configured route of every operation the binding runs is checked — and again at the attempt, and cannot reach the embedding call either. The dimensionality story is recorded as the code carries it: the route model holds the embeddings operation's two dimensionality properties — `dimensions`, the vector dimensionality the route declares, and `supports_dimensions_param`, the operator's statement that the route accepts a caller-selectable dimensionality — and the advertised dimensionality is the primary route's declared value. The rule applied to the caller's `dimensions` is stated as the code applies it: none stated sends none, one stated is forwarded as stated on a route that accepts the parameter, and refused with a capability error naming the parameter before any upstream call on one that does not — neither forwarded against the operator's configuration nor silently dropped. What is checked is the answer's shape, against the caller's stated dimension or, when none was stated, the route's declared one. (§9.6) The embeddings parameter column is corrected to what the code sends: `dimensions` travels when the caller states one and the route is configured to accept it. (§11.2) The embedding-space requirement is recorded: vectors are interchangeable only within one embedding space, two routes can declare the same dimensionality and produce incompatible vectors, so every route behind the embedding model must serve the same embedding space; the gateway enforces each answer's shape and each route's capability, and does not and cannot enforce the semantic agreement of two routes' vectors. (§4.7, §11.2) The classification of a structurally malformed upstream response is recorded as its own category — 502 Bad Gateway / gRPC `UNKNOWN`: refused on the route that answered wrongly, not retried on that route, but eligible for another, because the fault lies in that upstream's answer rather than in the request. A gateway programming defect stays an internal error, which remains neither retried nor walked.
- **1.20** — Records the embeddings endpoint as implemented (§4.7, §9.6). (§4.7) The endpoint is the documented subset of §4: the accepted request fields are `model`, `input`, `dimensions`, `encoding_format`, `user` and `metadata`, a string or an array of strings are both normalised to the ordered tuple the canonical request holds, and a key the gateway does not understand is refused with an invalid-request error naming it rather than dropped. The response is the documented shape — `object: "list"`, the public model ID, one `object: "embedding"` entry per input carrying its `index` and its `embedding`, and a usage object counting `prompt_tokens` and `total_tokens` — and the gateway's own accounting (invocation identifier, upstream model, attempt count) is deliberately not in it, because the OpenAI embeddings shape has no field for it. Input ordering is preserved and output dimensionality is verified at the gateway core: each vector is placed at the position the upstream reported, `data` is returned in the caller's input order whatever order the upstream used, a position set that is not exactly the caller's indices is refused with an internal error rather than misaligned, and every vector's length must equal the dimensionality the route declares or the caller asked for — a wrong-shaped vector is refused rather than returned, and a deployment that configures no dimensionality declares none and checks none. An unknown model is a model-not-found error reached before anything is attempted, a provider that cannot embed is refused with a capability error naming `embeddings`, and a route whose primary does not declare the capability is refused the same way. The endpoint is a supported OpenAI-compatible subset rather than drop-in compatible: the answer is the float form, so `base64` is refused with a capability error naming the encoding rather than answered in floats. (§9.6) The embeddings entry point is recorded as served under the same policy as the text one: the same single deadline, the same concurrency bound with the slot wait bounded by the remaining deadline, the same retry and fallback walk, and the same cancellation rule; it claims no exactly-once upstream execution, because `attempt_count` counts gateway attempts and the inference library's internal retry paths can carry more than one wire request inside one of them. The provider's capability is checked before anything is attempted — the model is resolved first, and only then is the provider asked whether it implements the embedding protocol.
- **1.19** — Records the revised terminal contract for a streamed answer as the implementation now enforces it (§16). The event vocabulary is five kinds, not four: `error` is the terminal event of a stream that failed after it had produced output, carrying the gateway's own stable reason and its sanitised message. The three outcomes are exhaustive and distinguishable by the consumer: a completed stream ends with exactly one `finish` event, always last, followed by `[DONE]`; a failed stream ends with exactly one `error` event, always last, and is closed without `[DONE]`; a truncated stream ends with no terminal event at all — client disconnect, transport loss, or an upstream that never reported a finish reason — and is closed without `[DONE]` either, so the terminator's presence or absence tells the three apart. Nothing may follow a terminal event and a stream produces at most one: the walk stops the moment it yields a `finish` or an `error`, and a failure that surfaces after a terminal event has been yielded is reported as the invocation's own error rather than as a second terminal event. A failure after output is no longer mislabelled as a finish: it is an `error` event carrying the gateway's own reason, because a finish reason classifies why generation stopped and must never carry a gateway reason — the previous shape borrowed `unknown` and parked the gateway's reason in the raw-reason field, overloading a field reserved for the provider's own and rendering a failure as a finish reason in the OpenAI-compatible shape. An upstream stream that ends without reporting a finish reason is classified as truncated rather than presented as a successful completion: the gateway synthesises no terminal event for an answer it never saw finish. The finish-reason claim about the inference library is corrected in both halves (§16, §9.6): the gateway preserves the provider's native reason whenever the library retains it in its native-reason field, on the streaming path exactly as on the non-streaming one, and only loses a reason the library normalised without retaining — LiteLLM maps an unrecognised provider reason to `stop` upstream of the gateway boundary, and that wording cannot be reconstructed downstream. The sequence-number claim no longer overstates: the numbers give ordering and reveal an internal gap, but they cannot by themselves prove that no trailing event was lost, because a stream that stops early shows no gap. The capability-refusal ordering is corrected: the model is resolved first, so an unknown model is refused as a model-not-found error whatever the bound provider can do, and the streaming-eligibility check follows, before any provider is called. The client-disconnect paragraph names the outcome it produces — a truncated stream, closed without `[DONE]` — and §11.2's first-visible-output paragraph is corrected to the `error` event the code emits (§11.2).
- **1.18** — A verification pass against the streaming implementation now landed, correcting the interface-grounded wording of §16 and the §11.2 first-visible-output paragraph. (§11.2) Both clauses of the boundary are now stated as the code enforces them: after the first delivered event the gateway makes no further attempt at all — not even a retry of the route that was streaming — and the failure is surfaced as the stream's terminal `finish` event, which carries the reason `unknown` and records the failure's own reason verbatim in its raw field; the failure is then raised, so the caller knows the invocation did not complete, and the partial output already delivered stands as the caller's answer. (§16) The capability refusal is corrected to what the code checks: streaming is a capability of the bound provider, not a route — it is absent from the route capability vocabulary and tested by no route check — so a provider that cannot stream is refused with a capability error naming `streaming` before the model is resolved and before any provider is called, rather than by the primary-first capability check a structured-output requirement triggers. The finish-event paragraph no longer promises a terminal event on every stream: a success produces one only where the upstream reported a finish reason and the gateway synthesises none, a refusal before the first event is an ordinary error response carrying no event, and the terminal event that reports a failure carries `unknown` with the failure's reason in its raw field rather than the failure itself. The raw reason on a successful stream is recorded as the inference library's normalised value — the library maps an unrecognised upstream finish reason to `stop` before the gateway sees the chunk — so an unrecognised reason cannot arrive on that path and no promise now depends on one doing so. The gRPC sentence is qualified: no gRPC streaming is implemented, so §16's guarantees are the HTTP ones. The client-disconnect paragraph states what the code does rather than nothing: the upstream work stops, the slot is released, and no terminal event is emitted, so a disconnected consumer sees a stream that stopped without its `finish` event rather than a failed one.
- **1.17** — Records the streaming contract as the canonical event model a stream is made of, with the guarantees a consumer can rely on (§16): the four event kinds — `content`, `tool_call`, `usage` and `finish` — and their ordering by a per-invocation sequence number that is 0-based and strictly increasing, so a consumer can tell events arrived in order and none was dropped; exactly one terminal `finish` event is produced for every stream, including one a failure ends, so a consumer can always distinguish a completed stream from a truncated one; a route that cannot stream is refused with a capability error naming `streaming` rather than answered with a complete, non-streamed body presented as incremental output; a stream is bounded by the same single deadline, the same concurrency bound and the same cancellation rule as a non-streaming invocation; and a client disconnect stops the upstream work. (§11.2) The first-visible-output boundary is recorded as an enforced rule rather than a stated expectation: once any event has been delivered to the caller, no fallback may occur, because a caller that has already seen output would receive a contradictory answer from a second provider; before the first event the ordinary retry and fallback rules apply unchanged, and a failure after it ends the stream instead of moving the invocation on.
- **1.16** — Separates the two eligibility decisions and bounds the two waits the reliability loop could otherwise spend unbounded. (§11.2) Retry eligibility and fallback eligibility are recorded as **different decisions**: a same-route retry test decides whether the route is attempted again, a separate fallback test decides whether the invocation moves on to the next eligible route, and a failure that answers no to both terminates the invocation where it happened. The categories are recorded per side: an unavailable provider and a deadline with budget left are both retryable and fallback-eligible; an upstream rate limit (`RATE_LIMITED`) is fallback-eligible but not retried on the same route; the deterministic request and caller failures — invalid argument, authentication, permission, missing resource or model, capability gap, failed precondition — an internal error, and the gateway's own concurrency refusal (`CONCURRENCY_LIMIT_EXCEEDED`) are neither. The rate-limit policy is stated: a rate limit moves the invocation to another route, because the taxonomy carries no retry-after value to wait the right amount and a blind retry would add load to the provider that just refused it. The backoff exponent is recorded as **route-local**, so a fallback's first retry backs off one base rather than inheriting the primary route's exponential growth. (§15) The wait for a concurrency slot is itself bounded by the invocation deadline, so an invocation that cannot be admitted before its deadline fails promptly instead of staying queued; the queue of waiting callers is bounded separately, and a full queue is rejected immediately with a resource-exhausted error rather than buffering unbounded work. The concurrency wording no longer promises strict arrival-order fairness, which the implementation does not guarantee. The retry backoff is recorded as jittered — extending a wait, never shortening it — with the randomness injectable so tests stay deterministic, and a backoff that would consume the remaining budget is refused with a deadline error rather than clamped and slept away.
- **1.15** — Records the retry, fallback, deadline and concurrency policy as the gateway core implements it. (§11.2) One invocation has one retry loop and one attempt count: a failure the gateway's own taxonomy classifies as retryable is retried on the **same route** within the configured `PROVIDER_RETRY_COUNT`, waiting a bounded exponential backoff derived from `PROVIDER_RETRY_BACKOFF_MS` — capped, clamped to the remaining budget, deterministic, and spent outside the concurrency slot — and a route whose attempts are used up hands the invocation to the next capability-eligible route. Retryability is decided from the gateway's categories and reasons, never from the inference library's exception types. The failure surfaced when no route could serve is the **last** attempt's, because that is the most informative, not the first. (§15) One deadline covers the whole invocation, so retries and fallbacks consume the same budget rather than each receiving a fresh one, and an attempt whose budget is exhausted is never started. The budget for one attempt is the smaller of the configured provider timeout and the remaining deadline. Simultaneous upstream attempts are bounded by `PROVIDER_MAX_CONCURRENT_REQUESTS`, and a concurrency slot is held only for the duration of an attempt and never across a backoff wait, so slow callers cannot starve others. Cancellation propagates without being converted into an upstream failure. `attempt_count` counts gateway attempts and must never be read as a wire-request count, because the inference library has an internal retry the gateway cannot suppress.
- **1.14** — Records what the binding actually enforces and narrows the claims around it. (§14.1) The two transport obligations hold on every outbound connection the verified dialects make, including the client the generic HTTP path's connection-error retry rebuilds — the binding hands that path the pinned transport itself, so the retry can no longer leave the controls; a dialect whose HTTP path has not been verified is refused at startup and again at the attempt rather than run behind controls that would silently not apply. Redirects are **refused, not evaluated**: a 3xx is neither followed nor assessed and surfaces to the caller as an upstream error, so the URL policy's redirect decision — a same-origin hop after revalidation, a cross-origin hop only by explicit opt-in, credentials never carried across an origin change — has no caller in a request path and controlled redirect handling is not yet implemented; §9 and §14 no longer claim the gateway evaluates each redirect. A network-level egress restriction is recorded as not required for the verified dialects, because pinning is enforced on every connection those paths make, and as the required compensating control only for a path outside the verified set, which the binding refuses rather than runs. (§6) The `.env` guard is **forced** rather than defaulted and is gateway-owned implementation state rather than operator configuration, so no ambient value — not even an explicit `LITELLM_MODE=DEV` — can re-enable the library's import-time load: configuration cannot arrive from the working directory, and the mode is not part of the gateway's public configuration surface. (§9.1) The response side is stated as the four-way classification the binding applies — a known canonical semantic maps, a known semantic with no canonical representation is refused, known non-semantic operational metadata is deliberately ignored from an explicit documented set, and an unknown provider-specific field is refused until it is classified — with the backend fingerprint (`system_fingerprint`) named as operational metadata that must not fail an otherwise valid response, and a recurring refusal named as evidence the canonical model needs extending rather than a reason to allowlist a semantic away. (§11.3) The internal connection-error retry is no longer a transport escape; what it leaves is an accounting consequence.
- **1.13** — Records three implementation realities alongside the obligations they qualify. (§14.1) The transport obligations are executable rather than aspirational: the binding injects a custom HTTP client into every upstream call — the inference library honours a per-call `client=` argument — that pins the connection to a validated address while preserving the original hostname for TLS SNI and the `Host` header, and that is built with redirect following disabled because the inference library hardcodes it on; omitting the injected client silently loses both controls, so it is not optional. (§11.3, §12) The retry reality: a zero retry count is mandatory on every call and suppresses both the library-level and the underlying SDK-level retry layers, so a normal invocation produces exactly one upstream request — but two internal retry paths, a message-reformat loop that fires on specific malformed-request responses and a connection-error retry on one code path that rebuilds a fresh client, cannot be switched off by any supported knob, so a reported `attempt_count` may understate the upstream requests by one and the gateway must not claim exactly-once upstream execution. (§6) Importing the inference library loads a `.env` file into the process environment — its DEV-mode import-time `dotenv.load_dotenv()` — which would let configuration appear outside the authoritative environment; the gateway sets `LITELLM_MODE` to a non-DEV value before the first import so the load never happens.
- **1.12** — Qualifies the inbound authentication guarantees and states the authentication scope policy (§7, §14.2): the `<key>:<tenant>` grammar forbids commas and colons in both API keys and tenant identifiers, because the list is comma-separated and the binding is colon-separated, and supporting opaque third-party credentials that contain those characters would require a structured representation; leading and trailing whitespace is trimmed and so cannot be part of a credential; the verification guarantee is stated accurately — the match position does not determine the number of comparisons, the number of configured credentials is not hidden and hiding it is deliberately not attempted because it is not a worthwhile secret, and comparisons run over precomputed fixed-length digests so no comparison is length-dependent; the log label is qualified as a correlation aid rather than a secret — operator-chosen credentials can be low entropy, so a logged prefix can be tested offline against candidate keys — and is distinguished from the credential's internal identifier, which downstream policy keys on and which need not appear in logs; and the authentication scope policy is stated precisely — all endpoints require authentication when it is enabled except health when `GATEWAY_ALLOW_ANONYMOUS_HEALTH` is true, a presented credential is validated even on an exempt path so a bad credential is never waved through, and caller-supplied `actor_id` and `agent_id` are attribution labels that must never become principals for quotas or authorization; only the credential's tenant and its internal credential identifier may serve as principals.
- **1.11** — Settles inbound authentication (§7): `GATEWAY_API_KEYS` is a comma-separated list whose entries are `<key>` or `<key>:<tenant>`, binding a key to a trusted tenant; the verification scan runs across every configured key, so timing does not reveal which key matched, while the number of configured credentials is not hidden and hiding it is deliberately not attempted; keys are plaintext in the operator's environment — this PRD's configuration model — and the gateway never logs or echoes a key, logging a derived non-secret key label instead; ambiguous key configuration is rejected at startup; the credential is accepted from `Authorization: Bearer` or `x-api-key`, with conflicting values rejected; and when authentication is enabled every `/v1/*` endpoint requires a valid credential, including the model listing because it exposes configured model IDs, with health endpoints exempt only when `GATEWAY_ALLOW_ANONYMOUS_HEALTH` is true. Also states at the canonical request's definition that `tenant_id`, `actor_id` and `agent_id` are caller-supplied and never authorization evidence, that trusted tenant identity derives solely from the validated credential, that a body `tenant_id` disagreeing with the authenticated tenant is refused with a permission error rather than silently ignored, and that an unauthenticated caller has no tenant, which a body value cannot create (§5.1).
- **1.10** — Tightens the capability-refusal rule to the primary route's advertised contract: a request is validated against the primary route's advertised capabilities first and refused when the primary does not declare the capability, even if a later route could honour it, because a fallback route may preserve a model's advertised capabilities but never silently expand them and the model listing advertises the primary route's capabilities (§4.3, §4.5, §6.1, §9.5); states the same rule in §11.2's fallback selection, where fallback candidates are filtered by the same capability test (§11.2).
- **1.9** — Adds `response_format` to the Responses API's supported fields (§4.5) together with the route requirement it carries: a non-text response format is honoured only by a route that declares the `structured-output` capability, and the invocation is refused with a capability error when the primary route's advertised capabilities do not declare it (§6.1, §9.5).
- **1.8** — Adds the `structured-output` route capability to the capability vocabulary (§4.3, §6, §18) and requires a route that cannot honour a structured-output requirement to be refused with a capability error rather than handed the request and allowed to answer with free text (§9.5); records that a system instruction is baseline chat semantics with no capability of its own, and that the gateway forwards an invocation-level instruction and a `system` message without merging either (§5.1, §9.5); and states the OpenAI compatibility boundary in §4 — the HTTP inference endpoints follow the compatible shapes for a supported subset rather than being drop-in compatible, and an unknown parameter is rejected as an invalid request rather than ignored, so a client SDK that serialises a new default parameter receives an error the caller can act on instead of a false assurance that its request was honoured.
- **1.7** — Closes a gap in the binding contract: adds §9.1, which requires every binding translator to account for every semantically relevant field of the upstream response and request — mapping it into the canonical model or rejecting the response as unsupported — states that rejecting unknown canonical keys is one layer of that defence rather than the enforcement point for upstream translation, and requires binding tests to use upstream fixtures carrying additional semantic fields so a drop cannot pass undetected; renumbers §9's remaining subsections and their cross-references. Also states at the canonical model's definition (§5.1) that request `metadata` is never forwarded to the model or provider unless a separately specified feature explicitly consumes a named key, so metadata cannot become an untyped provider-parameter bag.
- **1.6** — Replaces §8's stale path-joining prohibition with a rule against stripping, deduplicating, reordering or reinterpreting configured path components, so configured repetition is preserved; rewrites §14.1's network-level fallback as a compensating control that must block the prohibited address classes rather than claim equivalence to exact address pinning; and requires cross-origin redirects to be rejected by default unless a route opts in explicitly (§7, §14.1).
- **1.5** — Adds §14.1, which states that configuration-time URL screening alone is not SSRF prevention and requires the outbound transport to pin the connection to a validated address and to disable automatic redirect following, with a network-level egress restriction as the fallback where the binding cannot provide those controls; cross-references the obligation from §9 and narrows the SSRF wording in §9.2 and §14.
- **1.4** — Documents additive path joining in §8 (`*_PATH` is appended to the base path and never replaces it, so a provider prefix is never dropped) and sets `OTEL_ENABLED=false` in the §18 example to match the shipped environment templates.
- **1.3** — Corrections from architecture review: additional routes now carry their own authentication configuration plus only the credentials that configuration requires; drops the impossible duplicate-route-index rejection while keeping duplicate route identity and identical endpoint+model rejections; allows `*_EXTRA_HEADERS_JSON` on any route and replaces the unenforceable credential-injection wording with a reserved-authentication-header rule enforced at startup; tightens the §9 extensibility wording to "can be added through LiteLLM".
- **1.2** — Makes the LiteLLM boundary and ownership split explicit; collapses multi-route configuration to one schema (a singleton primary block plus `<CAPABILITY>_ROUTES_JSON`); assigns retry and fallback ownership solely to the gateway core with LiteLLM retry and fallback disabled; restricts an absolute-URL override in `*_PATH` to explicit opt-in.
- **1.1** — Removes grounded search and configurable model aliases; delegates all upstream inference to the LiteLLM library.
- **1.0** — Initial specification.

---

## 1. Executive Summary

The AI Model Gateway is a standalone service that provides a unified interface for interacting with artificial intelligence models from any compatible provider.

Applications communicate with the gateway using standardized gRPC or HTTP endpoints. The gateway handles model selection, request translation, provider authentication, inference execution, retries, fallbacks, usage tracking, and response normalization.

The gateway must be **vendor-neutral, configurable through environment variables, and independently deployable**.

A provider is defined by its endpoint, model identifier, authentication configuration, supported capabilities, and LiteLLM provider name.

The gateway delegates upstream protocol handling (request encoding, wire-format translation, and response decoding) to the LiteLLM library.

The gateway must not require a specific cloud provider, proprietary SDK, managed database, external orchestration service, or commercial AI vendor.

### Core design principles

1. **Vendor neutrality:** Provider selection is based on configured LiteLLM providers and capabilities, not vendor names.
2. **Protocol abstraction:** Applications use one consistent API regardless of the upstream protocol.
3. **Environment-driven configuration:** All deployment-specific settings, provider endpoints, model identifiers, credentials, and routing rules are configurable through environment variables.
4. **Standalone operation:** The gateway can operate without external application services.
5. **Optional infrastructure:** Persistence, caching, distributed coordination, and observability integrations are configurable.
6. **Security by default:** Authentication, credential protection, request validation, and network restrictions are first-class requirements.
7. **Deterministic behavior:** Retries, routing, errors, and fallback behavior must be predictable and testable.
8. **Extensibility:** Additional upstream providers can be added through LiteLLM without modifying the gateway's core business logic.

---

## 2. Product Goals

The gateway must provide:

- A unified gRPC API for model inference.
- An OpenAI-compatible HTTP API.
- Configurable text, image, and embedding providers.
- Multi-provider inference through the LiteLLM library.
- Provider-independent request and response representations.
- Support for multi-turn conversations.
- Tool/function calling with structured arguments.
- Multimodal input and output.
- Fixed logical model IDs with explicitly configured route fallback.
- Token usage and cost accounting.
- Configurable rate limits and concurrency limits.
- Optional API authentication and tenant isolation.
- Health checks and operational metrics.
- Container-ready deployment.
- Graceful startup, shutdown, and error handling.

### Non-goals for version 1

The following are not required for the initial release:

- Model training or fine-tuning.
- Hosting model weights or running inference engines internally.
- A graphical administration dashboard.
- A proprietary agent orchestration framework.
- Persistent conversation history.
- User billing or payment processing.
- Automatic model discovery across every provider.
- Arbitrary code execution from model-generated tool calls.
- Web search and grounded retrieval; applications integrate a search provider directly.

The gateway may integrate with external inference engines, but it does not implement the inference engine itself.

---

## 3. System Architecture

```text
                       CLIENT APPLICATIONS
                    ┌───────────┴───────────┐
                    │                       │
                gRPC API                HTTP API
                 :9090                    :8080
                    │                       │
                    └───────────┬───────────┘
                                │
                         API Translation
                                │
                         Canonical Request
                                │
                      Authentication / Policy
                                │
                        Model Route Resolver
                                │
                      Capability Dispatcher
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
          Text/Vision         Images           Embeddings
              │                 │                 │
              └─────────────────┼─────────────────┘
                                │
                              LiteLLM
                                │
                         Upstream APIs
```

All HTTP and gRPC operations must reuse common application services rather than duplicate inference logic.

---

# 4. Public HTTP API

The gateway must expose the following endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness |
| GET | `/v1/models` | List configured models |
| GET | `/v1/models/{model_id}` | Get model metadata |
| POST | `/v1/chat/completions` | Chat/text generation |
| POST | `/v1/completions` | Legacy text completion |
| POST | `/v1/responses` | Unified response generation |
| POST | `/v1/images/generations` | Image generation |
| POST | `/v1/images/edits` | Image editing |
| POST | `/v1/embeddings` | Text embeddings |

The inference endpoints should follow their commonly used OpenAI-compatible response shapes where applicable.

**Compatibility boundary.** That compatibility covers a supported subset of the OpenAI API; it is not drop-in compatibility with it. A request is honoured only when the gateway understands every parameter it carries, so an unknown parameter is rejected as an invalid request rather than ignored. A client SDK that serialises a parameter this gateway does not implement — most often a newly introduced default in a newer SDK release — therefore receives an invalid-request error, and the caller either removes the parameter or the gateway is extended to support it.

The rejection is deliberate. Silently accepting an unknown parameter would tell the caller its request was honoured while the gateway does not know what the parameter means. Such a parameter may change the result, and a gateway that ignored it would report an answer to a request it never fully understood; refusing it keeps the caller's view of the request and the gateway's view of it identical. It is the same unknown-key rule the canonical boundary applies everywhere (§9.1), applied to the wire format itself.

## 4.1 Health

`GET /healthz`

Returns HTTP 200 with:

```text
ok
```

Liveness must not depend on external provider availability.

## 4.2 Readiness

`GET /readyz`

Returns HTTP 200 when configuration is valid and the gateway can accept requests.

Returns HTTP 503 when mandatory internal dependencies are unavailable.

Provider outages should not automatically make the entire gateway unready when other configured routes remain usable.

## 4.3 Model Listing

`GET /v1/models`

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "text-default",
      "object": "model",
      "created": 0,
      "owned_by": "gateway",
      "context_window": 1048576,
      "max_output_tokens": 131072,
      "capabilities": ["chat", "tools", "vision", "structured-output"]
    },
    {
      "id": "embedding-default",
      "object": "model",
      "created": 0,
      "owned_by": "gateway",
      "capabilities": ["embeddings"],
      "dimensions": 1024
    }
  ]
}
```

The gateway must not expose upstream API keys, internal URLs, or other sensitive configuration.

## 4.4 Chat Completions

`POST /v1/chat/completions`

Request:

```json
{
  "model": "text-default",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain recursion."
    }
  ],
  "temperature": 0.7,
  "max_tokens": 2048,
  "stream": false
}
```

Response:

```json
{
  "id": "chatcmpl-example",
  "object": "chat.completion",
  "created": 0,
  "model": "text-default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Recursion is a technique..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 42,
    "total_tokens": 67
  }
}
```

The endpoint must support system, developer, user, assistant, and tool messages where the selected provider supports their semantics.

A `system` message is conversation-level. The invocation-level instruction a caller gives is a separate element — `instructions` on the Responses API (§4.5), `system_prompt` in the canonical request (§5.1) — and the gateway forwards both without merging either, because folding one into the other would change what the provider sees.

The request also accepts `idempotency_key` (§5.1), and the response carries `idempotency_status` beside `id` and `model` (§5.2). Those two are the exception to the absence of the gateway's own accounting from this payload, and they are the exception because a caller that cannot tell a first answer from a repeat of one cannot use the endpoint safely: the upstream model, the attempt count, fallback use and cost are still deliberately left out, because the chat completions shape has no field for them (§13).

**The legacy completion endpoint, `POST /v1/completions`, is implemented too.** It is served by the gateway core through the same text-generation operation as the chat endpoint, and its accepted request is its own documented subset: `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `stop`, `n`, `stream`, `echo`, `suffix`, `best_of` and `logprobs`. A key the gateway does not understand is refused with an invalid-request error naming it rather than dropped. A legacy completion has no roles, so `prompt` becomes one canonical user message and is forwarded verbatim — an empty prompt is honoured rather than refused — while an array prompt is refused with a capability error naming `multiple_prompts` rather than narrowed to its first element or joined into a prompt the caller never sent, because an array asks for one completion per element and the canonical invocation carries one conversation. The sampling parameters map onto the canonical generation config: `max_tokens` becomes the output bound the OpenAI name describes, and `temperature`, `top_p` and `stop` (a string or an array of strings) carry across.

**A recognised parameter this endpoint cannot honour is refused with a capability error naming the feature, and one whose value already matches gateway behaviour is accepted.** `n` above one asks for several completions and is refused naming `multiple_choices`, while `n: 1` is the gateway's own default; `logprobs` asks for token log-probabilities the canonical usage record does not carry and is refused naming `log_probabilities`, while `logprobs: false` is the upstream default; `echo` asks for the prompt to be echoed back and is refused naming `echo`, while `echo: false` is the upstream default; `best_of` above one asks for several samples with the best returned and is refused naming `best_of`, while `best_of: 1` is the upstream default; and `suffix` asks the model to continue a suffix after the prompt, which the canonical request has no field for, so any stated suffix is refused naming `suffix`, because no default value asks for nothing. A parameter whose value is the one the gateway already behaves like is accepted rather than refused, because rejecting it would reject a request the gateway can serve in full. A recognised parameter whose value is ill-typed for it — `{"n": true}`, `{"echo": 0}`, `{"best_of": "1"}` — is refused with an invalid-request error naming the parameter and the JSON type supplied, because a value the parameter's wire format does not define is unparseable input rather than a feature request, and an unvalidated default-equivalence test would read `{"n": true}` as `n: 1` and serve a request it silently reinterpreted.

**The response is the documented legacy shape, and the whole answer must fit it.** The response carries `id` (`cmpl-…`), `object: "text_completion"`, `created`, the public `model` ID, a single choice carrying the generated `text`, its `index` and its `finish_reason`, and `usage` counting `prompt_tokens`, `completion_tokens` and `total_tokens`. A token count the provider never reported is rendered as `0` for the same reason the Responses endpoint renders one: the legacy usage shape types every counter as an integer and has no null, so the gateway cannot tell 'the provider counted zero' from 'the provider never reported it' and reports zero rather than omit a field the shape requires. The gateway's own accounting — the invocation identifier, the upstream model, the attempt count — is deliberately not in this payload, because the legacy shape has no field for it (§5.2, §13). Two fields are the exception, for the reason the chat endpoint gives: the request accepts `idempotency_key` (§5.1), and the answer carries `idempotency_status` beside `id` and `model` (§5.2). The choice holds one `text` and nothing else, so a canonical answer carrying a tool call, a refusal or an image has no representation here: the invocation is refused with a capability error naming the feature — `tool_calls`, `refusal` or `image_output` — rather than returned without it, because the gateway does not report an answer delivered when part of it was dropped (§9.1, §9.5). A finish reason the gateway could not classify is reported verbatim rather than replaced by an invented OpenAI value, and `null` means the provider reported none.

**The endpoint is a supported OpenAI-compatible subset, not drop-in compatible.** The response shape is the compatible part; the request is validated rather than waved through. A parameter the gateway does not understand is rejected as an invalid request rather than ignored, so a client SDK that serialises a new default parameter receives an error the caller can act on instead of a false assurance that its request was honoured.

**Streaming is served as Server-Sent Events in the legacy chunk shape.** A request carrying `stream: true` is answered with a `text/event-stream` response whose chunks carry the fragment each canonical event reports under `choices[0].text`, with a usage report on a chunk carrying no choice and the finish reason on the terminal chunk with empty text. The stream's terminal contract is the one §16 defines: `[DONE]` is emitted only after a `finish` event, and a stream that ended with an `error` event, or with no terminal event at all, is closed without it, so a caller never reads a finished answer from a stream that simply stopped. The events come from the inference service's own stream, so routing, retry, fallback, the deadline and the no-fallback-after-first-output rule are the gateway core's, not duplicated here. A provider that cannot stream is refused with a capability error naming `streaming` before the response starts, rather than answered with a complete, non-streamed body (§9.5, §16). The legacy chunk shape has no place for a tool call, exactly as the non-streaming shape has none, so a tool-call event ends the stream with a terminal `error` event naming `tool_calls` and no `[DONE]`; a leading tool call, before anything has been delivered, is refused as an ordinary error response instead.

## 4.5 Responses API

`POST /v1/responses`

Supports:

- `model`
- `input`
- `instructions`
- `tools`
- `temperature`
- `top_p`
- `max_output_tokens`
- `tool_choice`
- `metadata`
- `stream` (accepted only as `false`; `true` is refused — see below)
- `response_format`

`response_format` is a structured-output requirement rather than an ordinary generation parameter. A non-text value — `json_object` or `json_schema` — is honoured only by a route that declares the `structured-output` capability (§6.1). The request is validated against the primary route's advertised capabilities first: when the primary route does not declare the capability, the invocation is refused with a capability error naming `structured-output`, even if a later route could honour it — a fallback route may preserve a model's advertised capabilities but never silently expand them (§9.5, §11.2). A `text` response format needs nothing beyond `chat`, because it is the provider's own default.

The gateway must normalize requests and responses into the Responses API representation.

A Responses-compatible upstream is not mandatory; a different LiteLLM provider may serve this endpoint when the requested semantics can be preserved.

Unsupported features must produce explicit errors rather than silently disappearing.

**The endpoint is implemented, and a Responses-compatible upstream is not required.** `POST /v1/responses` is served by the gateway core through the ordinary text-generation operation of §9.6: the canonical invocation is the one `POST /v1/chat/completions` builds and the canonical answer is the one the chat binding produces, so a provider that has never heard of the Responses API serves the endpoint without the gateway misreporting what it returned. The Responses representation is the caller-facing one, and rendering the canonical answer into it is the gateway's own work rather than the provider's.

**The accepted request is the documented subset, and nothing else is ignored.** The body carries `model`, `input`, `instructions`, `tools`, `temperature`, `top_p`, `max_output_tokens`, `tool_choice`, `metadata`, `stream` and `response_format`, and that set is closed: a key the gateway does not understand is refused with an invalid-request error naming it rather than dropped. `top_p` carries across onto the canonical generation config, `tool_choice` onto the canonical tool choice, and `metadata` onto the request metadata the invocation record already holds, which is bookkeeping and is never forwarded to the model or a provider (§5.2). A key the Responses format defines and this endpoint does not carry is refused with a capability error naming the feature it would request rather than dropped — `previous_response_id`, `text`, `truncation`, `reasoning`, `conversation`, `prompt` and the prompt-cache parameters, `max_tool_calls`, `moderation`, `service_tier`, `user`, `top_logprobs` and `parallel_tool_calls` among them — while a parameter whose value is the one the gateway already behaves like is accepted: `stream: false` asks for nothing, `store: false` matches a gateway that stores nothing, and the defaults `background: false`, `include: []` and `stream_options: {}` likewise ask for nothing. A recognised parameter whose value is ill-typed for it — `{"background": 0}`, `{"include": {}}`, `{"store": "yes"}`, `{"top_p": true}` — is refused with an invalid-request error naming the parameter and the JSON type supplied, because a value the parameter's wire format does not define is unparseable input rather than a feature request, and an unvalidated default-equivalence test would read a boolean as the integer that means 'one choice'. `parallel_tool_calls` is refused whatever its value, because neither `true` nor `false` is a distinction this gateway models. A nested key the Responses format defines and this endpoint cannot carry — `detail`, `file_id`, `logprobs`, `phase`, `prompt_cache_breakpoint` — is refused the same way, as is a tool of a non-`function` type, named by its type, and a function tool asking for `strict`, named `strict_tool_schemas`, because the canonical tool declaration carries no field for schema enforcement (§9.3, §9.5).

**The input is normalised into the canonical conversation.** A string `input` is the one user message it spells. An array carries message-like objects, each a role and an ordered content list, beside the two items a tool turn needs: `function_call` becomes the assistant message proposing the call, and `function_call_output` becomes the tool message answering it. A content part is `input_text`, `output_text` or `input_image`, the first two becoming canonical text and the third becoming a canonical image part the same way the chat endpoint's `image_url` part does: a base64 data URL is honoured, an `http`/`https` URL is refused with a capability error naming `remote_image_fetch` because the gateway does not fetch an image the caller could have sent, any other scheme is refused as malformed, and a route that does not declare the `vision` capability is refused naming it before the model is resolved. A replayed `output_text` part carrying a non-empty `annotations` list is refused with a capability error naming `annotations`, because the canonical content representation has no field for a citation. An input item or content part of any other type is refused with a capability error naming the type rather than skipped, because a dropped item changes the conversation the model sees (§9.1). A `tool` role on an input message is refused as an invalid request that names `function_call_output`, because a tool result is carried by the item that names the call it answers, and the input message carries no tool-call identifier.

**The answer is rendered content-first, then tool calls.** The Responses `output` array is ordered and mixes `message` items with `function_call` items, but the renderer emits every content part of the assistant turn first and then every tool call, so the order it can express is `(ContentPart)* (ToolCall)*` and an arbitrary interleaving of content and tool calls is not representable. One assistant turn's content parts become one `message` item's ordered content list, each part rendered as an `output_text` or a `refusal` variant, and each tool call becomes a `function_call` item after every content part. A tool-call-only answer carries no `message` item, exactly as the chat endpoint answers it with no content.

**Two canonical answers the representation cannot express are refused rather than approximated.** An answer carrying an image is refused with a capability error naming `image_output`, because a `message` item's content holds output text and refusals and nothing else. An answer whose content continues after a tool call is refused with a capability error naming `interleaved_output`: the wire format could carry the trailing content — nothing in the shape limits one `output` array to one `message` item — but the canonical model records ONE assistant turn, so emitting a second `message` item would invent a turn boundary the caller never sent, and joining the trailing content to the first would report "said did said" as "said said, did"; neither is the answer the model gave (§9.1, §9.5).

**The response is the documented subset of the Responses object this endpoint can state truthfully.** It carries `id` (`resp_…`), `object: "response"`, `created_at`, the public `model` ID, `status: "completed"`, the ordered `output` array, and a `usage` object counting `input_tokens`, `output_tokens` and `total_tokens` with an `input_tokens_details` object carrying `cached_tokens` and `cache_write_tokens` and an `output_tokens_details` object carrying `reasoning_tokens`. The two detail categories the canonical usage model does not track — cache-write tokens and reasoning tokens — are reported as zero rather than omitted, because the shape types them as integers. A token count the provider never reported is rendered as `0` for the same reason: the wire usage shape types every counter as an integer and has no null, so the gateway cannot distinguish 'the provider counted zero' from 'the provider never reported it' and reports zero rather than inventing an omission the shape has no field for. The gateway's own accounting — the upstream model, the attempt count, fallback use and cost — is deliberately not in this payload, because the Responses object has no field for it and adding one would be a contract the gateway invented (§5.2, §13); the fields that merely echo the request back — `instructions`, `tools`, `temperature` — are left out for the same reason. Idempotency is the one deliberate exception: the Responses object has no field for either, so this endpoint accepts no key and reports no status, and a caller deduplicates against the chat completions or legacy completions endpoint (§4.4) or the gRPC contract (§5.1, §5.2), not this one.

**Streaming is refused, explicitly, rather than faked.** `stream: true` is refused with a capability error naming `streaming`, before the model is resolved and before any provider is called, because the decision follows from the request alone. The Responses streaming protocol is an item-lifecycle protocol — it opens with `response.created` and `response.in_progress`, brackets every item and every content part with `added`/`done` events, streams text as `response.output_text.delta` and closes with `response.completed`, and every event carries an item id, an output index, a content index and a monotonic sequence number. The canonical stream (§16) carries none of that: its five kinds report a text fragment, a tool-call fragment, a usage snapshot, a terminal finish and a terminal error, with no item identity and no item boundary, so emitting the lifecycle events would mean the transport inventing item ids, item boundaries and event types the canonical vocabulary deliberately does not carry. The gateway will not answer the request with a complete, non-streamed body either, because that would tell the caller it received tokens as they were produced when it did not, so the request is refused. The endpoint therefore serves the non-streaming Responses representation only, which is a documented subset limitation.

**The endpoint is a supported OpenAI-compatible subset, not drop-in compatible.** The response shape is the compatible part; the request is validated rather than waved through. A parameter the gateway does not understand is rejected as an invalid request rather than ignored, so a client SDK that serialises a new default parameter receives an error the caller can act on instead of a false assurance that its request was honoured.

## 4.6 Images

`POST /v1/images/generations`

Example:

```json
{
  "model": "image-default",
  "prompt": "A futuristic city at sunset",
  "size": "1024x1024",
  "response_format": "b64_json"
}
```

Response:

```json
{
  "created": 0,
  "data": [
    {
      "b64_json": "<base64-encoded image>"
    }
  ]
}
```

**The endpoint is implemented.** `POST /v1/images/generations` is served by the gateway core through the image operation of §9.6, and the accepted request is the documented subset of §4: `model` (a public model ID), `prompt`, `size`, `response_format`, `n` and `metadata`. A key the gateway does not understand is refused with an invalid-request error naming it rather than dropped, and a request whose `prompt` is empty, whose `size` is outside the documented `WxH` form, whose `response_format` is outside the two documented wire representations — `url` and `b64_json` — or whose `n` is not a positive integer is refused the same way. The response is the documented shape: `created` is when the gateway assembled the answer, and each entry of `data` carries exactly one of `b64_json` or `url` — the two are alternatives in the shape this endpoint mirrors, so a caller reads whichever is present — beside `revised_prompt` when the upstream reported one. The gateway's own accounting — the invocation identifier, the upstream model, the attempt count — is deliberately not in this payload, because the OpenAI image shape has no field for it (§5.2, §13).

**The answer is checked against the request before it is returned.** A caller that stated `n` is answered with exactly that many images; an upstream that returned another number has not answered the request, so the invocation is refused rather than answered with the wrong number of images — a malformed-upstream-response error (502 Bad Gateway / gRPC `UNKNOWN`, §11.2), refused on the route that answered wrongly and eligible for another, because the fault lies in that upstream's answer rather than in the request. A caller who stated no count is checked against none: the upstream's own default applies, and no expected number exists. A result entry carrying both payloads, or neither, is refused at the gateway core rather than answered with a result a caller cannot read.

`POST /v1/images/edits` is served too, and its image input is validated where it arrives. The caller's image is accepted as a base64 payload, as a `data:<mediatype>;base64,<payload>` URL, or as an array of those — the representation this endpoint documents in place of multipart form data. An edit that carries no image at all, or whose payload is not decodable base64, is refused naming `image` rather than carried to a provider that would edit something else or fail on a corrupt body.

**Editing is refused with a capability error, for a reason the gateway owns.** An edit is an image invocation whose route must declare `editing` beside `generation` — the second being the operator's own statement (`IMAGE_GEN_CAPABILITIES`; §6.2). A route that declares `generation` alone is refused with a capability error naming `editing`, before any upstream call, and no later route can expand the advertised contract. The enforcement does not stop at the capability check, because the gateway owns a second reason: the canonical image request carries no field for an image input, so the caller's image cannot be forwarded to any route however its capabilities read, and an edit answered without it would report an image edited that never left the gateway (§9.5). A route that declares both capabilities is refused the same way, naming the same operation: the operation the gateway cannot honour is editing itself, and serving it awaits a canonical image request that can carry the image being edited.

**The endpoint is a supported OpenAI-compatible subset, not drop-in compatible.** The response shapes are the compatible part; the request is validated rather than waved through. A parameter the gateway does not understand is rejected as an invalid request rather than ignored, so a client SDK that serialises a new default parameter receives an error the caller can act on instead of a false assurance that its request was honoured (§8).

## 4.7 Embeddings

`POST /v1/embeddings`

```json
{
  "model": "embedding-default",
  "input": [
    "First document",
    "Second document"
  ]
}
```

Response:

```json
{
  "object": "list",
  "model": "embedding-default",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.12, -0.34, 0.56]
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

**The endpoint is implemented.** `POST /v1/embeddings` is served by the gateway core through the embeddings operation of §9.6, and the accepted request is the documented subset of §4: `model` (a public model ID), `input` (a string or an array of strings, both normalised to the ordered tuple the canonical request holds), `dimensions`, `encoding_format`, `user` and `metadata`. A key the gateway does not understand is refused with an invalid-request error naming it rather than dropped, and a request that is not a JSON object, or whose `input` is neither a string nor an array of strings, is refused the same way. The response is the documented shape: `object` is `list`, `model` is the public model ID that was asked for, each entry of `data` carries `object: "embedding"`, its `index` and its `embedding`, and `usage` counts `prompt_tokens` and `total_tokens`. The gateway's own accounting — the invocation identifier, the upstream model, the attempt count — is deliberately not in this payload, because the OpenAI embeddings shape has no field for it (§5.2, §13).

**Input ordering is preserved and output dimensionality is verified, at the gateway core.** The inference service places each vector at the position the upstream reported and returns `data` in the caller's input order whatever order the upstream used; the position set must be exactly the caller's indices, so an upstream that returned fewer, more or duplicated vectors is refused as a structurally malformed upstream response rather than answered with a vector matched to the wrong text. Every vector's length must equal the dimensionality the route declares (`EMBEDDINGS_DIMENSIONS` on the route, the value the model listing advertises for the primary) or, when the caller asked for one, the dimensionality the caller asked for; a vector of the wrong length is refused rather than returned. Both refusals are classified as a malformed upstream response, the gateway's own category for an answer that cannot serve a valid request (502 Bad Gateway / gRPC `UNKNOWN`, §11.2): refused on the route that answered wrongly and not retried on that route, but eligible for another, because the fault lies in that upstream's answer rather than in the request — another eligible route may well answer the same request correctly. A deployment that configures no dimensionality declares none, and none is checked — an unconfigured limit is not a limit of zero (§4.3, §6.3).

**An unknown model is a model-not-found error.** The public model ID is resolved before anything is attempted: a request naming a model the gateway does not expose is refused with a model-not-found error (404) whatever the bound provider can do, so a caller-supplied model name cannot reach an unconfigured upstream endpoint (§10). A provider that cannot embed is refused with a capability error naming `embeddings`, and a route whose primary does not declare the `embeddings` capability is refused the same way — a fallback route may preserve the advertised capabilities but never expand them (§9.5).

**The endpoint is a supported OpenAI-compatible subset, not drop-in compatible.** The response shapes are the compatible part; the request is validated rather than waved through. A parameter the gateway does not understand is rejected as an invalid request rather than ignored, so a client SDK that serialises a new default parameter receives an error the caller can act on instead of a false assurance that its request was honoured. The gateway's embeddings answer is the float form: `encoding_format` of `float` — or no stated encoding, which is the upstream's own default — is honoured, and `base64` is refused with a capability error naming the encoding rather than answered in floats (§9.5).

---

# 5. gRPC API

Default port: `9090`.

The gateway must define its own versioned protobuf contract.

Package:

```protobuf
package gateway.v1;
```

Service:

```protobuf
service ModelGatewayService {
  rpc Invoke(InvokeRequest) returns (InvokeResponse);
  rpc InvokeStream(InvokeRequest) returns (stream InvokeChunk);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}
```

The service must also implement the standard gRPC Health Checking Protocol, served on the same port, using the `grpcio-health-checking` package's own implementation rather than a clone.

Regenerating the stubs from the contract is a command, not an edit — the generated files are the compiler's output and are excluded from linting for that reason:

```
mkdir -p .gen
uv run python -m grpc_tools.protoc -I proto --python_out=.gen --grpc_python_out=.gen proto/gateway/v1/gateway.proto
mv .gen/gateway/v1/gateway_pb2.py src/model_gateway/grpc/gateway_pb2.py
mv .gen/gateway/v1/gateway_pb2_grpc.py src/model_gateway/grpc/gateway_pb2_grpc.py
rm -rf .gen
```

The stub's import of the pb2 module must then be rewritten from `from gateway.v1 import gateway_pb2` to a relative `from . import gateway_pb2` — grpcio-tools emits the package-qualified form, which does not resolve inside `model_gateway/grpc/`, and regeneration silently breaks the package without the fixup.

The listener starts and stops with the application (§18), so one process serves both transports under one lifetime, and both are configured by the same environment: `CHORA_GRPC_PORT` (§6) is the port, and the listener binds every configured model's routes the same way the HTTP listener does.

The health protocol is served **without authentication**, deliberately. It is the infrastructure protocol — load balancers, service meshes and orchestrators probe it, and a prober presents no API key and has none to present. It answers one of three serving states and carries no gateway data. The HTTP health endpoints, by contrast, follow the deployment's configured auth policy. The distinction is intentional and is recorded here so a future change does not "fix" it into an inconsistency: HTTP health is the gateway's application endpoint, gRPC health is an infrastructure one.

## 5.1 InvokeRequest

The canonical gRPC request must represent:

| Field | Type | Purpose |
|---|---|---|
| `invocation_id` | string | Unique invocation identifier |
| `tenant_id` | string | Optional tenant identity |
| `actor_id` | string | Optional caller identity |
| `agent_id` | string | Agent/service identifier |
| `model` | string | Public model ID |
| `messages` | repeated message | Conversation |
| `system_instruction` | string | The invocation-level instruction |
| `generation_config` | structured object | Model parameters |
| `tools` | repeated message | Tool declarations |
| `tool_choice` | message | How the model may use the declared tools |
| `response_modality` | enum | Text, image, multimodal |
| `fallback_models` | repeated string | Permitted fallback routes |
| `idempotency_key` | optional string | Caller-supplied deduplication key |
| `metadata` | bytes | Non-sensitive request metadata, as canonical-domain JSON text |

`tenant_id`, `actor_id` and `agent_id` are caller-supplied identity fields, and they are never authorization evidence. Trusted tenant identity derives solely from the validated credential (§7): a key configured as `<key>:<tenant>` carries its tenant from the operator's configuration, and no request field can grant, widen or change that tenant. A request-body `tenant_id` that disagrees with the authenticated tenant is refused with a permission error rather than silently ignored. When the caller is unauthenticated there is no tenant at all, and a body `tenant_id` cannot create one.

Request `metadata` is bookkeeping for the caller — routing hints, correlation and provenance attributes, telemetry annotations. It is never forwarded to the model or to the upstream provider. Metadata reaches an upstream call only when a separately specified feature of this gateway explicitly names a key and defines its meaning; absent such a specification, no metadata key is copied into the request the binding constructs. Metadata must not become an untyped bag of provider parameters: forwarding caller-supplied keys into `litellm.completion`, `litellm.embedding` or `litellm.image_generation` would give an inbound caller control over parameters the canonical contract does not type, and would let a provider-specific key override gateway-owned policy.

`system_instruction` is the invocation-level instruction, and it is not interchangeable with a `system`-role message in `messages`: the first instructs the invocation, the second is part of the conversation the caller supplied, and the gateway forwards both without merging either (§9.5). An instruction needs no route capability of its own, because it is baseline chat semantics rather than an optional behaviour: the route's `chat` capability carries it, and a binding that cannot map an instruction must reject the invocation as unsupported rather than drop it (§9.1). The gateway's own contract names the field for the canonical one: `system_prompt` is not vendor-specific enough to justify the divergence, and a field name is the kind of thing that is expensive to change after publication.

`tool_choice` is the caller's tool-choice policy, the canonical one: `AUTO` leaves the decision to the provider, `NONE` forbids calling the declared tools, `REQUIRED` obliges the model to call one, and `SPECIFIC` names the one it must call. A `SPECIFIC` choice naming a tool the invocation does not declare is refused, because it is not a request any provider could honour. Sending no choice at all is a different request from sending `NONE`, and the contract keeps the two apart.

`idempotency_key` is the caller's own deduplication key, and the field is `optional` so that a request stating no key and one stating an empty key stay distinct across the boundary. The canonical record draws that line — an absent key is `None`, and an empty one is refused rather than read as no key at all — and a plain `string` would collapse the two into a single empty value. The key names the deduplication rather than the request — it is no part of the request's fingerprint — so a retry carrying a freshly minted invocation identifier is still the same request (§12).

### Canonical fidelity

The contract is deliberately the canonical model rather than a provider dialect, so a field is chosen for what the canonical type can express and never for the provider's convenience. That draws one line, and every field falls on one side of it:

- **A field the canonical model types is typed protobuf.** Scalars, enums and structured messages the gateway validates are protobuf's own: a model ID, a generation bound, a modality, a usage count.
- **A field the canonical model holds as arbitrary JSON is canonical-domain JSON bytes.** Metadata, a tool's parameters, a JSON Schema and a tool call's arguments are `bytes` carrying the JSON text, which the mapping parses and enforces the shape the canonical type requires.

The line exists because of what the obvious alternative does. The only JSON-ish well-known protobuf types represent every number as a double, so a JSON integer arrives as a float and one beyond 2^53 is lost: `1` becomes `1.0`, and a schema's numeric constraint can change value before it reaches the canonical model. Canonical JSON text carries an integer exactly.

Two consequences worth stating:

- **A JSON Schema may be a boolean.** `true` and `false` are valid JSON Schemas, so the schema field is JSON text and the mapping accepts an object or a boolean and refuses every other kind — mirroring the canonical invariant exactly.
- **An absent raw finish reason stays absent.** The canonical record distinguishes "the provider reported no reason" (`None`) from "the provider reported an empty one" (`""`), and the field is `optional string` so the two stay distinct across the boundary rather than collapsing into one empty string. The canonical `StreamEvent` draws the same line; it had not, and the streaming path could not preserve the distinction the gateway's own transport requires.

The protobuf schema must assign stable field numbers and preserve backward compatibility after publication.

## 5.2 InvokeResponse

Required concepts:

- Invocation ID.
- Public model ID.
- Resolved upstream model ID.
- Text content.
- Tool calls.
- Optional image content.
- Finish reason.
- Token usage, **including which counts the gateway derived rather than read** — the usage message carries a closed set of field names marking the counts the gateway estimated, so an estimated count is reported as an estimate and not as a provider fact, and a field the provider never reported is reported as absent rather than as zero (§13).
- Estimated cost.
- Provider attempt count.
- Whether a fallback was used.
- Idempotency status.

The gateway's own finish reason classifies why generation stopped and the provider's own reason is recorded verbatim beside it; the two are kept apart, and an absent raw reason (`None`) is a different fact from an empty one (`""`), which the contract preserves (§9.1).

The idempotency status is one of four values, and it is the one piece of gateway accounting a caller needs in order to use the endpoint safely, because it says whether the provider was reached. `not_applicable` — the invocation carried no idempotency key, so deduplication was never in play. `new` — the key was accepted and this invocation is the first answered under it. `replayed` — the key matched an earlier invocation with the same request fingerprint, and the recorded answer to that invocation is what the caller is reading; the provider was not called again, which is what prevents duplicate accounting. `indeterminate` — the gateway cannot say which of those it did, which is the honest answer after a timeout against a provider with no idempotency support (§11.3). A key reused for a different payload is a second request rather than a status: it is answered as `new`, and the first request's answer is never given for it (§12).

## 5.3 Streaming

`InvokeStream` must use server-side streaming.

Chunks must include:

- Invocation ID.
- Sequence number.
- Content delta or tool-call delta.
- Optional usage information, carrying the same estimate provenance §5.2 requires.
- Terminal finish reason, carrying the same absent-versus-empty distinction.
- Error details where appropriate.

A stream that cannot serve the request is refused before the first chunk with the gateway error's own status and no chunk at all; a failure after the first chunk is reported by exactly one terminal error chunk, after which the stream ends — the event is the failure's report and the raise that follows it is the control flow that ends the stream, never a second failure (§16).

Streaming is optional at the upstream level only if the gateway can explicitly reject it. The gateway must not pretend a non-streaming provider response is token-by-token streaming.

## 5.4 Error codes

Use standard gRPC status codes:

- `INVALID_ARGUMENT`
- `UNAUTHENTICATED`
- `PERMISSION_DENIED`
- `NOT_FOUND`
- `RESOURCE_EXHAUSTED`
- `FAILED_PRECONDITION`
- `DEADLINE_EXCEEDED`
- `UNAVAILABLE`
- `INTERNAL`

Error messages must be sanitized and contain stable machine-readable reasons.

---

# 6. Environment Configuration

Environment variables are the authoritative deployment configuration.

All variable names are uppercase and use underscores.

Configuration is loaded and validated at startup.

**The working directory cannot contribute configuration.** Importing the inference library loads a `.env` file into the process environment: the library's DEV-mode default calls `dotenv.load_dotenv()` at import time, and its file search finds a `.env` in the working directory on interactive launches and otherwise walks the directory tree upward from the library's own location — which passes through the project directory in this layout. Every variable the file carries is copied into `os.environ`, so an endpoint, a credential or a model ID the operator never set in the deployment environment becomes effective gateway configuration. The gateway prevents the load by **forcing** `LITELLM_MODE` to `PRODUCTION` — an assignment, not a default — in gateway-owned implementation state, at the one seam through which the gateway imports the inference library and before the first import of it. Because the value is forced rather than defaulted, no ambient variable can disable the invariant: an operator who exports `LITELLM_MODE=DEV` does not re-enable the load, and the process environment stays the only configuration source. Configuration therefore cannot arrive from the working directory, and `LITELLM_MODE` is not part of the gateway's public configuration surface — the gateway neither reads it as configuration nor documents it as an operator variable, and its only other effect is on the inference library's proxy CLI, which this gateway does not run. A process that imports the inference library outside that seam, before any gateway module, is outside this guarantee, which is why the gateway imports the library only through the seam.

An absent optional capability is disabled rather than preventing unrelated capabilities from operating.

`*_FORMAT` keeps its name in every capability block. Its value is the provider dialect name understood by the inference binding — a LiteLLM provider name such as `openai` or `vertex_ai`, not the company operating the endpoint. The operation performed (chat, embedding, image generation) is determined by which capability block the variable belongs to, not by the variable name: `TEXT_LLM_FORMAT` selects the dialect for chat and text, `EMBEDDINGS_FORMAT` for embeddings, and `IMAGE_GEN_FORMAT` for image generation.

`*_CAPABILITIES` names the behaviours a route declares, drawn from one shared vocabulary: `chat`, `tools`, `vision`, `structured-output`, `generation`, `editing`, and `embeddings`. Which values are meaningful follows from the operation the block configures: a text route uses `chat`, `tools`, `vision` and `structured-output`; an image route uses `generation` and `editing`; an embeddings route uses `embeddings`. A declared capability is a claim the gateway relies on when it selects a route, so it must not be declared for a provider that does not support it: the gateway would then send a request the route cannot honour, having checked the very capability that was supposed to prevent exactly that.

## 6.1 Text generation

```dotenv
TEXT_LLM_ENDPOINT=https://api.openai.com/v1
TEXT_LLM_MODEL_ID=gpt-sunburst
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_PATH=
TEXT_LLM_EXTRA_HEADERS_JSON={}
TEXT_LLM_CAPABILITIES=chat,tools,vision,structured-output
TEXT_LLM_CONTEXT_WINDOW=
TEXT_LLM_MAX_OUTPUT_TOKENS=
```

`TEXT_LLM_MODEL_ID` is the bare upstream model name; the gateway composes `<format>/<model_id>` when calling LiteLLM.

Supported formats (LiteLLM provider names):

| Format | Provider |
|---|---|
| `openai` | OpenAI and OpenAI-compatible endpoints |
| `anthropic` | Anthropic Messages API |
| `openrouter` | OpenRouter |

A listed format is runnable only because the binding has verified that the inference library's HTTP path for it honours the injected client, which is what carries the transport controls of §14.1 — address pinning and disabled redirect following. The verified set today is exactly `openai`, `anthropic` and `openrouter`; a `TEXT_LLM_FORMAT` outside it is refused at startup, and again at the attempt, rather than run with the controls lost (§14.1).

The other LiteLLM provider names of the initial build are not yet verified, and a route naming one is refused at startup: `vertex_ai` (Vertex AI), `gemini` (Gemini API), `ollama` (Ollama) and `azure` (Azure OpenAI). Their HTTP paths have not been verified to carry the transport controls, and a path that discards the injected client loses both controls with no error to notice. Each becomes runnable once its path is verified and added to the verified set.

`TEXT_LLM_FORMAT` is the provider dialect name for chat and text understood by the inference binding — one of the verified set above, a LiteLLM provider name, not the company operating the endpoint.

`structured-output` is what a text route must declare to serve a caller's non-text response format — `json_object` or `json_schema` (§9.5). The request is validated against the primary route's declared capabilities: when the primary route does not declare it, the invocation is refused with a capability error naming `structured-output`, even if a later route declares it — a fallback route may preserve the primary route's capabilities but never expand them. A `text` response format asks for nothing beyond `chat`, because it is the provider's own default.

## 6.2 Image generation

```dotenv
IMAGE_GEN_ENDPOINT=
IMAGE_GEN_MODEL_ID=
IMAGE_GEN_API_KEY=
IMAGE_GEN_FORMAT=openai
IMAGE_GEN_AUTH_TYPE=bearer
IMAGE_GEN_PATH=
IMAGE_GEN_EXTRA_HEADERS_JSON={}
IMAGE_GEN_CAPABILITIES=generation
```

`IMAGE_GEN_FORMAT` is the provider dialect name for image generation, and `IMAGE_GEN_MODEL_ID` is the bare upstream model name.

**The image generation entry point is implemented, and an image dialect is runnable exactly when its own HTTP path is verified.** `POST /v1/images/generations` is served through `litellm.aimage_generation` (§9.6), the image counterpart of the text entry point: a different library call reached through the same per-attempt client seam, so the transport obligations of §14.1 travel with the call rather than with the operation. The image entry point is nevertheless a different library entry point from the text binding's — its dispatch is not the text one's, and on the generic dialects it serves a synchronous path — so each dialect's image path was verified on its own against the pinned library, and the runnable set that verification established is `openai`, which takes the OpenAI SDK path, and `gemini` and `vertex_ai`, which share the generic path. The dialect refusal happens before any client is built, on that shared seam, so an unverified dialect cannot reach the image call either: it is refused at startup, where every configured route of every operation the binding runs is checked, and again at the attempt, exactly as for text and embeddings (§6.1, §6.3, §14.1).

The provider names this block was initially built around that remain outside that set are refused rather than run, each for the reason its image path's verification established (§9.7): `azure`, whose image path silently discards the injected client and reaches the wire on its own, and `anthropic` and `openrouter`, which the library's image dispatch does not route to an image answer at all. Each becomes runnable once its image path is verified and added to the set, exactly as for the text formats.

Image editing is enabled only when explicitly supported: `IMAGE_GEN_CAPABILITIES` accepts `generation` and `editing`, and a route declaring `generation` alone must not be handed an edit request (§4.6). `structured-output` is a text-route capability and is not meaningful for an image block, whose response carries image data rather than a schema-conforming document.

## 6.3 Embeddings

```dotenv
EMBEDDINGS_ENDPOINT=
EMBEDDINGS_MODEL_ID=
EMBEDDINGS_API_KEY=
EMBEDDINGS_FORMAT=openai
EMBEDDINGS_AUTH_TYPE=bearer
EMBEDDINGS_PATH=
EMBEDDINGS_DIMENSIONS=
EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM=false
EMBEDDINGS_EXTRA_HEADERS_JSON={}
EMBEDDINGS_CAPABILITIES=embeddings
```

`EMBEDDINGS_FORMAT` is the provider dialect name for embeddings, and `EMBEDDINGS_MODEL_ID` is the bare upstream model name.

**The embeddings entry point is implemented, and its runnable formats are the binding's verified set.** `POST /v1/embeddings` is served through `litellm.aembedding` (§9.6), the embeddings counterpart of the text entry point: a different library call reached through the same per-attempt client seam, so the transport verification the text formats carry is not a separate question the embeddings path had to answer again. An embedding dialect is runnable exactly when its HTTP path is in the verified set of §14.1 — `openai`, which takes the OpenAI SDK path, and `anthropic` and `openrouter`, which share the generic HTTP path. The dialect refusal happens before any client is built, on that shared seam, so an unverified dialect cannot reach the embedding call either: it is refused at startup, where every configured route of every operation the binding runs is checked, and again at the attempt, exactly as for text (§6.1, §14.1). The provider names this block was initially built around that remain outside that set — `vertex_ai`, `gemini`, `azure`, `ollama` — are refused rather than run, and become runnable once their HTTP paths are verified and added to the set, exactly as for the text formats.

`EMBEDDINGS_CAPABILITIES` defaults to `embeddings`, the only capability an embeddings route offers.

The gateway must support both fixed-dimension and configurable-dimension models.

**Dimensionality is a property of the route, and a stated caller dimension is forwarded only on a route the operator configured to accept it.** The route model carries the embeddings operation's two dimensionality properties: `dimensions`, the vector dimensionality the route declares, and `supports_dimensions_param`, the operator's own statement that this route accepts a caller-selectable dimensionality. Both are set per route — `EMBEDDINGS_DIMENSIONS` and `EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM` on the singleton block, or `dimensions` and `supports_dimensions_param` on an additional route — and a declared dimensionality must be a positive integer, because a vector of zero or negative length is not a dimensionality any model has. The model listing advertises the primary route's declared dimensionality (§4.3), and a route that declares none checks none: an undeclared dimensionality is not a dimensionality of zero. The rule the gateway applies to the caller's `dimensions`: no caller dimension sends none, and the upstream's own default applies; one stated is forwarded to the upstream as stated, but only on a route the operator configured to accept the parameter. On a route that does not accept it, the request is refused with a capability error naming the parameter, before any upstream call — the parameter is neither forwarded against the route's configuration nor silently dropped. The refusal is validated against the primary route's contract, and no later route can expand it: a fallback that does not accept the parameter is skipped rather than attempted when the request states one (§9.5, §9.6). What the gateway checks is the answer: every vector's length must equal the caller's stated dimension, or the route's declared dimensionality when the caller stated none, and a wrong-shaped vector is refused rather than returned (§4.7, §9.6).

---

# 7. Authentication

Supported upstream authentication mechanisms:

| Value | Behavior |
|---|---|
| `bearer` | Authorization bearer token |
| `x-api-key` | `x-api-key` header |
| `api-key` | `api-key` header |
| `none` | No upstream authentication |
| `custom-header` | Configurable authentication header |

The gateway decides which credential configuration applies to a route and supplies it to the binding; LiteLLM builds the provider's native authentication header from what it is given. Precedence:

1. **Native provider authentication is the default.** The gateway supplies `api_key` for the route and normally lets LiteLLM construct the provider's native header — for example the OpenAI-compatible `Authorization: Bearer` header.
2. **`*_AUTH_TYPE` applies only when a custom endpoint needs a non-native header.** It exists for endpoints that are not the provider's native endpoint and require `x-api-key`, `api-key`, or a named custom header. A value that matches the provider's native scheme (for example `bearer` for an OpenAI-compatible endpoint) is a no-op and equivalent to leaving the variable unset.
3. **`*_EXTRA_HEADERS_JSON` may be used on any route.** It carries non-authentication headers a route legitimately needs — provider routing headers, organisation or project identifiers, proxy-required headers, and feature or version headers — and is independent of the route's authentication configuration.

Extra headers merge with the headers LiteLLM generates for the request. The gateway's configured headers are applied last, so a configured header name wins over a LiteLLM-generated header with the same name, and LiteLLM's remaining generated headers are preserved.

Reserved authentication headers must not be supplied through `*_EXTRA_HEADERS_JSON`: `Authorization`, `x-api-key`, `api-key`, and any header the route's own authentication configuration owns. Supplying a reserved header requires an explicit custom-auth configuration for the route; otherwise startup must reject the configuration. The reserved-header rule is the enforceable form of the constraint, because the headers a route's authentication owns follow from its `*_AUTH_TYPE` and credential configuration and can therefore be checked at startup, whereas "must not inject a credential" describes an intent that no configuration check can decide.

Credentials are bound to one origin. The gateway must never forward a route's credentials to a different origin, and must never re-send them across a redirect that changes scheme, host, or port. A cross-origin redirect is rejected by default for that reason, and is permitted only through an explicit per-route opt-in (§14.1).

For `vertex_ai` routes, upstream Google credentials are resolved by LiteLLM through Google Application Default Credentials, and the gateway passes no static key. ADC is ambient: on GCE or GKE the workload's attached service account is inherited automatically, so operators must scope that service account deliberately.

For example:

```dotenv
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_API_KEY=example-secret
```

Produces:

```http
Authorization: Bearer example-secret
```

The gateway must separately configure inbound client authentication. Upstream credentials must never be accepted from ordinary inference request bodies.

**Key format.** `GATEWAY_API_KEYS` is a comma-separated list. Each entry is either `<key>`, which authenticates the caller without binding it to a tenant, or `<key>:<tenant>`, which binds that key to one trusted tenant. A tenant established this way is trusted because the operator's configuration established it — not because the caller asserted it (§5.1). The grammar constrains its parts: neither an API key nor a tenant identifier may contain a comma or a colon, because the list is comma-separated and the binding is colon-separated — a comma would split one credential into two entries, and a second colon makes the entry unreadable, which startup rejects. Supporting opaque third-party credentials that contain those characters would require a structured representation rather than this flat grammar. Leading and trailing whitespace is trimmed from every entry and from both halves of a binding, so whitespace can never be part of a credential.

**Verification.** The presented credential is hashed once to a fixed-length digest and compared against every configured credential's precomputed digest; the scan runs to the last credential whether or not an earlier one matched. The match position therefore does not determine the number of comparisons, and because every comparison is between two fixed-length digests, the comparison work does not depend on the length of the presented credential or on ordinary content equality between the digests. The number of configured credentials is not hidden: total work still scales with it. Hiding it is deliberately not attempted, because the count is not a worthwhile secret — it says nothing about the credentials themselves and an attacker cannot act on it. A presented credential that matches nothing behaves identically to a key that was never configured.

**Plaintext keys.** Keys are plaintext in the operator's environment, which is this PRD's configuration model (§6). The gateway never logs or echoes a key, and logs a derived non-secret key label instead, so an operator can tell which credential was presented without the key itself appearing in logs or error responses. The label is a correlation aid, not a secret: it is a short hash prefix, and operator-chosen credentials can be low entropy — the example keys in this PRD are guessable — so a logged prefix can be tested offline against candidate keys until one hashes to it. Credentials must therefore be generated with high entropy, and the label must never be treated as a secret. The label is distinct from the credential's internal identifier, which downstream policy — rate limiting, concurrency limits, idempotency and accounting — keys on as the principal naming the credential. That identifier is internal gateway identity: it is never exposed in public API responses or ordinary logs, while trusted internal persistence and policy components may store and use it — durable accounting may write it to a database, distributed rate limiting may carry it through a shared store, and tracing or audit records may include it — and it need not appear in logs. It is an unsalted deterministic digest, so, like the label but with definitive verification rather than prefix matching, possessing it permits offline candidate testing; it is therefore sensitive internal data, though it is never itself accepted as authentication material. A keyed derivation would avoid that exposure, at the cost of key-management and rotation obligations that are not warranted now.

**Startup validation.** Ambiguous key configuration is rejected at startup: a key that carries conflicting tenant bindings — appearing both unbound and bound, or bound to two different tenants — is a configuration error, because the gateway must not choose between bindings by an arbitrary rule.

**Presentation.** The credential is accepted from the `Authorization: Bearer` header or the `x-api-key` header. Conflicting values are rejected: when both headers are present and disagree, the request is refused rather than resolved by a precedence rule.

**Endpoint coverage.** When authentication is enabled, every `/v1/*` endpoint requires a valid credential, including the model listing (§4.3), because the listing exposes configured model IDs. The health endpoints (§4.1, §4.2) are exempt only when `GATEWAY_ALLOW_ANONYMOUS_HEALTH` is true.

Recommended inbound configuration:

```dotenv
GATEWAY_AUTH_ENABLED=true
GATEWAY_API_KEYS=primary-key,internal-tooling:tenant-acme
GATEWAY_ALLOW_ANONYMOUS_HEALTH=true
```

Production must reject unauthenticated inference requests when authentication is enabled.

---

# 8. Provider URL Resolution

The gateway must support both plain base URLs and base URLs that include an explicit path.

Example:

```dotenv
TEXT_LLM_ENDPOINT=https://provider.example/v1
TEXT_LLM_FORMAT=openai
```

Resolved base URL handed to the inference binding:

```text
https://provider.example/v1
```

With an explicit path:

```dotenv
TEXT_LLM_PATH=/custom/generate
```

Resolved base URL handed to the inference binding:

```text
https://provider.example/v1/custom/generate
```

A relative path in `*_PATH` is appended to the base path and never replaces it, so a configured provider prefix such as `/v1` is never silently dropped. That is what satisfies the prohibition below on silently stripping meaningful path components, and it is what makes `*_PATH` useful now that the operation path is appended by the inference binding.

A full URL in `TEXT_LLM_PATH` must not override the base URL by default. Deterministic joining governs: the value is treated as a path and appended to the base URL. An absolute-URL override requires an explicit opt-in for the route, for example `TEXT_LLM_ALLOW_ABSOLUTE_PATH_URL=true`; without that opt-in, an absolute URL in `*_PATH` is invalid configuration and startup must reject it rather than resolve it silently.

URL joining must be deterministic.

The gateway must not silently strip, deduplicate, reorder, or otherwise reinterpret configured path components. Repetition produced by the literal configuration is preserved: an endpoint of `https://provider.example/v1` with `TEXT_LLM_PATH=/v1/custom` resolves to `https://provider.example/v1/v1/custom`.

The resolved value is handed to the inference binding as its base URL. LiteLLM appends operation paths (for example `/chat/completions`) to that base URL; those operation paths are LiteLLM's responsibility, not the gateway's.

---

# 9. LiteLLM Binding

The gateway binds to LiteLLM as its vendor-neutral inference layer. Each configured route supplies a LiteLLM provider name, an upstream model name, a base URL, and an API key, and the gateway calls LiteLLM with a provider-prefixed model string.

LiteLLM owns provider-specific request encoding, network execution, and response decoding for text, vision, tool calling, image generation, and embeddings. That is the whole of the boundary: **LiteLLM owns upstream protocol translation and execution, and nothing else.**

The gateway owns policy, lifecycle, validation, identity, reliability policy, and the public contract. It also re-normalises LiteLLM's output at its own boundary: a LiteLLM result is treated as upstream data, mapped into the canonical representation, and checked against the fields the gateway depends on, never forwarded to callers unchanged.

In particular, LiteLLM does not own error normalisation or authentication decisions. It may raise provider-shaped exceptions, and it may construct a provider-native authentication header from credentials the gateway supplies (§7), but the gateway decides which error category a caller sees and which credential configuration applies to a route.

Because the binding is what opens the outbound connection, it also carries the transport obligations that make the gateway's URL policy effective: the connection must be pinned to an address the gateway validated, and automatic redirects must be disabled, so a redirect is never followed automatically (§14.1). The binding enforces both, and it does not yet evaluate a redirect it is handed — a redirect is refused rather than assessed (§14.1). Configuration-time screening in the gateway and connection-time control in the transport are two halves of one requirement, and neither alone prevents SSRF.

This satisfies the extensibility principle: additional upstream providers can be added through LiteLLM without changing the gateway's core business logic.

## 9.1 Upstream translation and semantic preservation

Strict canonical parsing — rejecting unknown keys — protects the HTTP/gRPC boundary and every persisted representation. It does not protect the provider translation boundary. When a binding constructs a canonical response from a LiteLLM object, a provider field dropped on the floor is invisible to that parser, because the parser never sees the upstream object.

Every binding translator must account for every semantically relevant field of the upstream response and request: each such field is either mapped into the canonical model, or the response is rejected as unsupported. A translator must not silently discard an upstream field because no canonical target exists for it today. Dropping a field silently changes what the caller was told — a model that returned a refusal, a cached-content flag, a log-probability block, a provider-specific stop reason or a content-safety annotation stops reporting that fact, and the gateway cannot surface a difference it never observed.

**The response side is a four-way classification, and the binding applies it rather than assuming it.** Every field the binding inspects falls into exactly one class:

- a **known canonical semantic** is mapped into the canonical model;
- a **known semantic the canonical model cannot represent** is refused, with the field named — log probabilities, reasoning content, audio, a legacy function call, a second choice, a usage-detail count with no canonical target — because the caller must not be told the answer was honoured when part of it was dropped;
- **known non-semantic operational metadata** is deliberately ignored, from an explicit documented set: a provider diagnostic that does not change the answer must not fail an otherwise valid response once it is understood;
- an **unknown provider-specific field** is refused until it is classified, so a new provider annotation cannot pass unexamined.

Classification is an act, not a default. A field enters the ignored set only when it is understood to be operational metadata rather than answer semantics, and the reason is recorded beside it; listing a field is a decision that it changes nothing about the answer, never a place to park a semantic the canonical model cannot hold. The backend fingerprint (`system_fingerprint`) is the field this applies to today: it identifies the upstream infrastructure that produced the answer, not anything about the answer — two backends serving the same model return the same tokens — so it is operational metadata, it is ignored, and **it must not fail an otherwise valid response**. The upstream's own identity fields — `id`, `created`, `model` — need no entry in that set: the gateway stamps its own identity on the response (§5.2), so an upstream identity is deliberately neither mapped nor refused.

A refusal that recurs is evidence about the canonical model, not a licence to widen the ignored set. A repeated refusal of, say, an extra token category in a usage breakdown is a signal that the canonical usage model (§13) may genuinely need a field for that category, because a token count is a reported fact about the answer rather than backend metadata. Until the canonical model carries it, the refusal stands: allowlisting a semantic away would hide a difference the caller was never told about.

The same rule holds on the request side. Every semantically relevant element of a canonical request must be represented in the upstream call, or the invocation is rejected as unsupported. An upstream capability the canonical request asks for and the provider cannot honour must produce a structured capability error (§9.5), not a quietly degraded call.

Rejecting unknown canonical keys is one layer of this defence, not the point at which upstream translation is enforced. Unknown-key rejection runs at the canonical boundary on data the gateway itself defines; the preservation rule runs on the upstream boundary on data the provider defines. Neither substitutes for the other: a binding that rejects unknown canonical keys and drops provider fields has satisfied the first rule and violated the second.

Binding tests must prove the preservation rule holds. They must use upstream fixtures carrying additional semantic fields — fields beyond the minimum a happy-path response needs — so a translator that drops them fails the test. A fixture containing only the fields the canonical model already captures cannot detect a drop.

## 9.2 Binding responsibilities

The binding must not own:

- Tenant authorization.
- Global routing.
- Billing policy.
- Request idempotency.
- Global retry policy.
- Audit persistence.
- Error normalisation.
- Authentication decisions.
- The public response contract.

These responsibilities belong to the gateway core.

## 9.3 Tool calling

The canonical representation must support:

- Function name.
- JSON-schema parameters.
- Tool-call identifier.
- Tool arguments.
- Tool result.
- Tool-choice policy.

LiteLLM translates tool definitions and calls to and from the upstream provider's format.

**The gateway must not execute arbitrary model-requested tools.** The caller or an explicitly registered trusted tool executor remains responsible for execution.

## 9.4 Vision

The canonical content representation must support text and images in the same message.

Image inputs may be supplied as:

- Base64 content.
- Authorized remote URLs.
- Supported binary payloads.

Vision content is passed through to LiteLLM, which handles provider-specific encoding.

Remote image fetching must satisfy the same controls as upstream calls: the gateway screens the URL against its policy, and the fetch is performed under the transport-level controls in §14.1 (address pinning and no automatic redirect following), together with a size limit.

## 9.5 Unsupported features

When a provider does not support a requested capability, the gateway must return a structured capability error.

It must not silently discard tools, images, system instructions, structured-output requirements, or other semantically important parameters.

**Structured output.** A structured-output requirement is a capability in its own right, not a generation parameter the provider may treat as advice. A caller's non-text response format — `json_object` or `json_schema` — is honoured only by a route that declares the `structured-output` capability (§6). The request is validated against the primary route's advertised capability contract first: the model listing advertises the primary route's capabilities (§4.3), so when the primary route does not declare `structured-output` the invocation is refused with a capability error naming it — even when a later route declares it. Serving the request from that later route would answer a request the public contract never advertised for the model: a fallback route may preserve a model's advertised capabilities but never silently expand them (§11.2). Answering with free text while the caller believes its output shape was guaranteed is the failure this forbids, and the caller's schema — not the provider's convenience — decides whether the requirement was met. A `text` response format needs nothing beyond `chat`.

**System instructions.** A system instruction needs no capability of its own, because it is baseline chat semantics rather than an optional behaviour: a route that declares `chat` claims to handle an instruction, and a binding that cannot map one must refuse the invocation rather than drop it or fold it into the conversation. The gateway carries an instruction in the form it was given. An invocation-level instruction — `system_prompt` in the canonical request (§5.1), `instructions` on the Responses API (§4.5) — and a `system` message inside `messages` are separate elements, and the gateway forwards both without merging either, because silently folding one into the other would change what the provider sees (§9.1).

## 9.6 Route-to-invocation mapping

Each capability resolves to exactly one LiteLLM entry point. The gateway passes the resolved base URL as the endpoint, the route's API key, and a provider-prefixed model string (`<format>/<model_id>`). The endpoint is always passed as a base URL; operation paths are appended by LiteLLM (§8).

| Capability | LiteLLM entry point | Parameters passed |
|---|---|---|
| Chat / text | `litellm.completion` (streaming: `stream=True`; async: `litellm.acompletion`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `messages`, generation parameters, `timeout` drawn from the invocation deadline, `num_retries=0` |
| Embeddings | `litellm.embedding` (async: `litellm.aembedding`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `input`, `dimensions` when the caller states one and the route accepts it, `num_retries=0` |
| Image generation | `litellm.image_generation` (async: `litellm.aimage_generation`) | `model` = `<format>/<model_id>`, `api_base` = resolved endpoint, `api_key`, `prompt`, only the image parameters the route's dialect and model accept (§9.7), `num_retries=0` |

Retries are disabled on every call because the gateway core owns the retry loop (§11.3).

**The embeddings entry point is served under the same policy as the text one.** `InferenceService.embed` resolves the public model ID to the routes configured for the embeddings operation, validates the request against the primary route's capability contract — which for this operation means the `embeddings` capability, and only that — and then runs the same walk: one deadline covers the invocation, every attempt is made inside a concurrency slot whose wait is bounded by what is left of that deadline, a retryable failure is retried on the same route within the configured count, a fallback-eligible failure moves to the next eligible route, a failure that answers no to both terminates the invocation where it happened, and the failure reported when nothing can serve is the last attempt's (§11.2, §11.3, §15). Cancellation propagates: a cancelled embeddings invocation stops at the attempt in flight, releases the slot that attempt held, and makes no further attempt. The embeddings path makes no exactly-once claim: `attempt_count` counts the calls the gateway makes, and the inference library's internal retry paths mean one of those calls can carry more than one upstream request (§11.3, §12).

**The provider's capability is checked before anything is attempted.** The model is resolved first — an unknown ID is a model-not-found error whatever the bound provider can do — and only then is the provider asked whether it implements the embedding protocol; a provider that cannot embed is refused with a capability error naming `embeddings` rather than handed the request and left to improvise (§9.5). A caller who asked for a wire encoding the gateway cannot produce is refused the same way, before any provider is called: the canonical answer is the float form, so `base64` is a capability error rather than an answer in floats (§4.7).

**A finish reason the library kept is the provider's own.** LiteLLM maps recognised finish reasons onto its OpenAI-shaped vocabulary and keeps the provider's original in its native-reason field whenever the two differ; the binding reads the native reason from that field when it is present and the mapped reason otherwise, on a stream exactly as on a non-streaming call, and the gateway's own vocabulary classifies whatever that was. A reason the library normalised without retaining cannot be recovered downstream: LiteLLM maps an unrecognised provider reason to `stop` before the gateway boundary, so the gateway records the value it was handed — classified with its own vocabulary — and the provider's own wording is lost upstream of the gateway, not deleted by it.

**"Supported in this gateway" is a narrower claim than "supported somewhere in LiteLLM".** A provider appearing in LiteLLM's catalogue is not acceptance evidence for this gateway. A provider is supported here only when a configured route names it *and* the gateway's own tests cover its operation, custom endpoint, authentication, and response semantics. LiteLLM may know how to call a provider this gateway has never validated, and the gateway must not advertise or route to one.

## 9.7 Image generation constraints

The supported upstream operation is `POST {endpoint}/images/generations`. A route advertises image generation only when that operation is what its endpoint provides.

**Which image dialects are runnable is a property of the image entry point's own verification.** The image entry point is a different library entry point from the text and embedding ones, so none of their verification carries over: the image dispatch is not the text one's, and each candidate dialect's image path was driven against the pinned `litellm-core` with a recording transport. The verified set is `openai`, which takes the OpenAI SDK path, and `gemini` and `vertex_ai`, which share the generic path — served *synchronously*, so the client the binding injects there is a synchronous `HTTPHandler` over a synchronous pinned transport, not the `AsyncHTTPHandler` the text path's generic dialects take. A dialect outside that set is refused, at the attempt and at startup, before any client is built, each for the reason its path's verification established:

- `azure` silently discards the injected client — a recording transport inside the injected client captures nothing while the upstream is still reached — so the call runs on the library's own client, which re-resolves the hostname and follows redirects with nothing raised: the one failure mode §14.1 exists to prevent.
- `anthropic` is not in the library's image dispatch at all: the call answers without a wire request, an empty 200 that is worse than a raised error.
- `openrouter` is dispatched, but its image path does not produce the image-answer shape: the call reaches the upstream and comes back with an empty `data` list — an image answer carrying no images.

**The transport controls hold on this entry point.** Address pinning and disabled redirect following reach the wire through the injected client on both supported paths: with the injected client, a 302 from the upstream is not followed — it surfaces as an error, an internal error on the SDK path and a connection failure on the generic one — and only because the injected client refuses it, since without the injection both paths follow the redirect themselves. `num_retries=0` travels on every call, as on every call the binding makes, because the gateway core owns the retry loop (§11.3) — and the image entry point is the one call that does not read it: the library pops `max_retries` off the request data (default 2) and assigns it onto the injected SDK client, so a plain injected SDK client is forcibly retried twice — measured, a 500-ing upstream saw three wire requests inside one gateway attempt. The binding's retry-pinned SDK client keeps the count at zero, so one gateway attempt carries exactly one wire request on a supported route; the generic path's synchronous handler makes exactly one wire request too, having no connection-error retry of its own.

**Image support must not be overstated: the parameters a route can accept are its dialect's and model's own.** The library raises `UnsupportedParamsError` for a request parameter its provider configuration does not list, before any request is made — measured for a model the configuration does not recognise, with `response_format` on every candidate dialect. The binding asks the same configuration object the library will use — for the bare model ID, because that is the string the library's dispatch hands to it — and forwards only what that set contains: a stated `n` or `size` the configuration does not accept is refused with a capability error naming it, before any upstream call, rather than forwarded to the library's own opaque refusal. `response_format` has one measured honour-by-default case: every verified dialect that does not list the parameter produces base64 image data by construction — the Google Imagen transformations emit `inlineData` and `bytesBase64Encoded`, and the OpenAI GPT configuration's upstream defaults to `b64_json` — so a caller who asked for `b64_json` is honoured by the upstream's own single representation and the parameter is not forwarded, while a caller who asked for `url` on such a route is refused with a capability error naming `response_format`. A refusal the library raises anyway is classified as the capability refusal it is, not as an unavailable upstream.

---

# 10. Model Routing

The gateway exposes a fixed set of public model IDs. They are part of the public contract and are not configurable aliases:

```text
text-default
image-default
embedding-default
```

Each public model ID maps to a configured capability route and to that route's configured fallbacks (§11).

An unknown public model ID must return a model-not-found error.

Caller-supplied model names must not allow access to unconfigured upstream endpoints.

The word "alias" is reserved for configurable mappings. The gateway defines none: a public model ID is a fixed identifier, and callers cannot introduce, rename, or remap one.

---

# 11. Multiple Providers and Fallbacks

The gateway must support multiple routes for a capability. Fallbacks are required: a capability may declare a primary route and any number of additional routes.

## 11.1 Configuration schema

There is exactly one schema. For each capability, the singleton block defines the primary route and a `<CAPABILITY>_ROUTES_JSON` variable supplies the additional routes as a JSON array.

```dotenv
TEXT_LLM_ENDPOINT=https://primary.example/v1
TEXT_LLM_MODEL_ID=primary-model
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai

TEXT_LLM_ROUTES_JSON=[{"endpoint":"https://fallback.example/v1","model_id":"fallback-model","api_key":"","format":"openai"}]
```

The same pattern applies to every capability: `IMAGE_GEN_*` with `IMAGE_GEN_ROUTES_JSON`, and `EMBEDDINGS_*` with `EMBEDDINGS_ROUTES_JSON`.

How the singleton block and `*_ROUTES_JSON` combine:

- The singleton block always defines the primary route (index 0). It is the only place the primary route is defined.
- `*_ROUTES_JSON` entries are additional routes, appended in array order (index 1, 2, ...).
- An absent or empty array means the capability has exactly one route.
- Precedence is positional, not field-merged. An entry never overrides a field of the singleton block, and no field is inherited from one route to another: each entry is self-contained and must specify its own endpoint, model ID, format, and authentication configuration, plus any credentials that configuration requires. Credentials are never carried across routes, because a fallback route may point at a different origin (§7).
- A capability is enabled only when its singleton block is configured. `*_ROUTES_JSON` alone, with the singleton block unset, is ambiguous and must be rejected at startup.

The indexed form (`TEXT_LLM_1_*`, `TEXT_LLM_2_*`) does not exist. It is not a supported configuration, and its presence must be rejected at startup.

Ambiguous configuration must be rejected at startup rather than resolved by guessing. This includes a malformed JSON array, an entry missing a required field, duplicate route identity (two entries presenting themselves as the same route), and two entries naming the same endpoint and model.

## 11.2 Retry and fallback policy

One invocation has one retry loop and one attempt count, and the loop is the gateway core's (§11.3). The primary route is attempted first. Every failure is asked two separate questions — *may another attempt on this route help?* and *may another route help?* — and the answers decide whether the route is retried, whether the invocation moves on to the next eligible route, or whether it terminates where it happened. Route order is fallback order, so the walk is the primary route, then the configured fallbacks in order.

```text
Primary
   ↓ retryable on this route, retries remain
Primary retry
   ↓ route used up, failure fallback-eligible
Fallback 1
   ↓ route used up, failure fallback-eligible
Fallback 2
   ↓ no route left
Last attempt's failure
```

**Retries apply to the same route and are bounded by the configured count.** `PROVIDER_RETRY_COUNT` is how many times one route may be retried after its first attempt, so a route is attempted at most `PROVIDER_RETRY_COUNT + 1` times. A failure a repeat cannot change is not retried at all; whether the walk then moves on to another route or the invocation ends there is the fallback-eligibility question below.

**Retry eligibility and fallback eligibility are two different decisions, not one.** *May another attempt on this route help?* is the same-route retry test; *may another route help?* is a separate fallback test, and the two answers differ for the same failure. The retry test decides whether the route is attempted again within `PROVIDER_RETRY_COUNT`; the fallback test decides whether, once the route is used up, the invocation moves on to the next eligible route. A failure that answers no to both terminates the invocation where it happened, and that failure is what the caller sees — the gateway does not spend another provider's capacity on a request that cannot succeed, and does not let the caller's answer depend on which providers happen to be configured.

Both tests are decided from the gateway's own error taxonomy — the category the failure was classified into and, where the category carries several reasons, the reason — and from nothing else. A failure that has not been classified into the taxonomy has no gateway meaning yet, so it is neither retried nor walked. The inference library's exception types play no part: the gateway classifies every failure into its own categories and reasons first (§5.4), and both decisions read those categories and reasons, never the type the library raised.

Which side each category falls on:

- **Retryable on the same route, and fallback-eligible:** an unavailable provider (`PROVIDER_UNAVAILABLE`) — the route could not serve the request, so both a repeat and a different route may succeed — and a deadline that still has budget left, since another attempt, here or elsewhere, may fit inside what remains.
- **Fallback-eligible but not retryable on the same route:** a rate limit (`RATE_LIMITED`) and a structurally malformed upstream response (`MALFORMED_RESPONSE`, 502 Bad Gateway / gRPC `UNKNOWN`). The limit belongs to the route that answered it, and another route is not necessarily limited, so the invocation moves on; the same route is not retried (see the rate-limit policy below). A malformed response is the same shape of decision for a different reason: the request was valid and the route was reachable, but the answer was structurally wrong — the wrong number of vectors, a position set that is not the caller's, a vector of the wrong dimensionality, a record the canonical representation cannot carry — so the fault lies in that upstream's answer. The route that just answered wrongly is not retried, because a repeat is unlikely to answer differently, but another eligible route may well answer the same request correctly, so the invocation walks. A malformed response is its own category rather than an internal error, because widening that category would make every gateway defect fallback-eligible, and a defect is not something another route can fix (§4.7).
- **Neither retryable nor fallback-eligible:** the deterministic failures that describe the request or the caller — an invalid argument, an authentication or permission failure, a missing resource or model, a capability the route does not advertise, a failed precondition — and an internal error. Another provider cannot change the answer to an invalid request, so the invocation terminates immediately rather than being retried on the same route or walked to the next; an internal error is a gateway defect, which §15 classes as a deterministic error category, reported rather than retried into. The gateway's own concurrency refusal (`CONCURRENCY_LIMIT_EXCEEDED`) is here too: it is the gateway's limiter refusing work, which every route shares, so neither a repeat nor another route would be admitted any differently.

**The rate-limit policy: a rate limit moves the invocation to another route; it is not retried on the same one.** An upstream rate limit is not retried on the route that answered it, because the gateway's taxonomy carries no retry-after value to wait the right amount, and retrying blind would add load to the provider that just refused the request. It is not discarded either — it stays fallback-eligible, because the limit belongs to that route and another route may not be limited. The gateway's own `CONCURRENCY_LIMIT_EXCEEDED`, raised by the concurrency limiter when its queue is full, is treated differently again: it is neither retried nor walked, because every route shares the same limiter and would refuse the invocation identically.

**A retry waits a bounded, jittered exponential backoff.** The delay after a failed attempt grows exponentially from `PROVIDER_RETRY_BACKOFF_MS` — one base after the first attempt, two after the second, four after the third — capped at a fixed ceiling so no single wait can dominate an invocation. The exponent is **route-local**: it counts the retries this route has already made, not the invocation's total attempts, so a fallback's first retry backs off one base exactly as the primary route's first retry did, rather than inheriting exponential growth from the attempts an earlier route already spent. The delay carries jitter, and a delay that would consume the invocation's remaining budget is refused rather than slept away (§15). The wait is spent outside the concurrency slot, and it is waited only when a retry will actually follow it: the last attempt a route is allowed never sleeps for a retry that will not be made.

**When a route is used up and the failure is fallback-eligible, the next capability-eligible route is tried.** A route is used up when its retries are exhausted, when the failure is one a repeat cannot change, or when a backoff that could not fit the remaining budget ended its retries. The walk continues only when the fallback test says another route may help; a deterministic request or caller failure answers no to both tests and terminates the invocation instead. A fallback route may preserve a model's advertised capabilities but never expand them. Fallback candidates are the later routes that could serve the request unchanged: each is filtered by the same capability test the primary route was validated against, so a route that does not declare everything the request needs is skipped rather than attempted, and no invocation is ever served a capability the model listing does not advertise (§4.3, §9.5).

**Embedding vectors are interchangeable only within one embedding space, and that is a property of the configuration, not something the gateway can verify.** Two embedding models can both return vectors of the same dimensionality — the same declared length, the same float encoding, one vector per input — and still produce vectors that are not comparable: nothing in a wire answer reveals which embedding space a vector belongs to, and the gateway has no way to test two spaces against each other. A fallback between different embedding models can therefore return a syntactically valid answer that is semantically incompatible with what the first route would have returned, which for anything that persists or compares vectors is worse than a failure: the wrongness surfaces later, in the caller's own records, with nothing in the answer to reveal it. Every route configured behind the embedding model must therefore serve the same embedding space. What the gateway enforces is the shape of each answer and the capability of each route: a fallback route is held to the same dimensionality check as the primary, and a route that does not declare `embeddings` is skipped rather than attempted (§4.7, §9.5). What it does not enforce is the semantic agreement of two routes' vectors — dimensionality equality is necessary and nowhere near sufficient, and a deployment that configures two embedding spaces behind one model gets exactly the fallback it configured.

**The failure surfaced when nothing can serve is the last attempt's, not the first.** The most recent failure is the one that says why the invocation could not be served, so that is the error the caller sees; a deadline that ran out before another attempt could begin is surfaced as a deadline error instead (§15).

Same-route retries may be triggered by:

- Connection failures.
- Timeouts.
- HTTP 500–599.
- Equivalent transient provider errors.

Each is a retry trigger once it has been classified into a category the gateway retries on the same route — an unavailable provider, or a deadline that still has budget left; an upstream condition the gateway does not classify as retryable on the same route does not trigger one. An upstream rate limit (HTTP 429) is deliberately absent: it is a fallback trigger, not a same-route retry trigger (see the rate-limit policy above).

**Once an event has been delivered, no fallback may occur.** The first event delivered to the caller is a boundary the retry loop enforces. Before it, the ordinary rules above apply unchanged: a retryable failure retries the same route, and a fallback-eligible failure walks to the next eligible route, exactly as for a non-streaming invocation. After it, no fallback may occur, because a caller that has already seen output from one provider would receive a contradictory answer from a second — two providers' completions of the same request are not interchangeable once either has been shown. A failure after the first delivered event therefore ends the stream instead of moving the invocation on: the gateway makes no further attempt, on this route or another — not even a retry of the route that was streaming — and the failure is surfaced as the stream's terminal `error` event (§16), which carries the gateway's own stable reason and its sanitised message rather than a finish reason, because the provider did not finish. The partial output the caller has already received stands as the caller's answer, and the failure is then raised, so the caller knows the invocation did not complete.

## 11.3 Retry and fallback ownership

The gateway core owns retries and fallback. LiteLLM's own retry and fallback support is disabled, so that there is exactly one retry loop, one deadline, and one attempt count for an invocation. The loop's mechanics — same-route retries bounded by the configured count, taxonomy-based retryability, the bounded backoff, the walk to the next eligible route, and the last attempt's failure reported — are specified in §11.2.

- LiteLLM is called with retries disabled (`num_retries=0`). The gateway must not rely on LiteLLM's `fallbacks` or `context_window_fallbacks` parameters.
- **The zero retry count is mandatory on every call, and it suppresses both retry layers.** `num_retries=0` switches off LiteLLM's own retry loop and the underlying SDK client's retries together, so in the normal case an invocation produces exactly one upstream request. Injecting a client does not remove the obligation: a call that passes a custom SDK client without the zero count has that client's retry default silently restored, so the count travels with the call, whatever else the call carries.
- **Two internal retry paths cannot be switched off.** No supported knob disables a message-reformat loop that rewrites and resends the request when the upstream answers specific malformed-request responses, or a connection-error retry on one code path that rebuilds its client for the second attempt. The rebuild no longer escapes the transport controls — it is built on the pinned, non-redirecting transport the binding supplies (§14.1) — so what remains is an accounting consequence: the gateway counts the calls it makes, and one gateway attempt can carry two upstream requests, so a reported `attempt_count` may understate the number of upstream requests by one under those conditions. The gateway must not claim exactly-once upstream execution (§12).
- The gateway enforces a single total deadline per invocation (§15). Provider timeouts and connect timeouts are budgeted inside that deadline, and no retry or fallback may extend it.
- Retries must never switch providers after visible streaming output has been emitted to the caller.

Stacked retry loops are prohibited: neither the gateway core nor the LiteLLM binding may wrap a second retry layer around a call that the other already governs.

---

# 12. Idempotency

Every invocation may carry an idempotency key, and the gateway must prevent duplicate accounting for equivalent repeated requests. A repeat of the same request is answered from the gateway's own record of the first answer rather than by asking the provider again, so the provider is called once and every equivalent repeat is served from the record.

The mechanism is implemented in the inference service, and four rules shape it. Each is a reason the obvious implementation is wrong.

**The fingerprint covers the payload, not the key.** A key names a caller's intent to deduplicate; it says nothing about what was asked. The record therefore stores a fingerprint of the request — the model, the conversation, the instruction, the generation config, the tools and the tool choice, the modality, the permitted fallbacks and the tenant the request acts under. The fingerprint is taken over a canonical JSON encoding of that content, never over `repr()`, because a dataclass's repr is an implementation representation that changes when the class does, which would invalidate every stored fingerprint and couple the contract to Python's formatting.

**A key names one operation, so reuse is a conflict rather than a re-run.** Once a key has a record, a repeat carrying a different fingerprint is refused with `idempotency_conflict` (409 Conflict / gRPC `ALREADY_EXISTS`) rather than executed as a new request. Executing it would run a request the caller never made and overwrite the record the key already holds, so one key could oscillate between unrelated payloads indefinitely. The conflict is refused before any provider is called and leaves the stored record intact.

**The gateway-minted invocation identifier is not part of the fingerprint.** The gateway mints its own identifier and discards a provider's, so a retry carrying the same key and a freshly minted identifier is the same request. Fingerprinting the identifier would make every retry a miss and defeat the mechanism.

**Only a completed answer is recorded.** A failure is not cached, because a retry after a transient failure should be allowed to retry, and recording a failure as if it were an answer would tell the caller the invocation succeeded. The consequence is that the gateway claims no exactly-once upstream execution: a provider without idempotency support, or a timeout that leaves the outcome unknown, may produce more than one upstream call, and the inference library's internal retry paths can issue a second upstream request inside a single gateway attempt (§11.3).

**The key is namespaced by the trusted principal, not used raw.** The store is keyed by a namespace derived from the authenticated credential — its identifier and its tenant — beside the caller's key. Two tenants using the same key are two independent operations, and letting one overwrite the other's record is a cross-tenant interference channel: after the overwrite the first tenant has lost its replay. The namespace is derived from the credential by the transport, because a tenant is a property of the credential and never of the request (§5.1). A credential with no tenant is still namespaced by its identifier, so two tenant-less credentials do not share a namespace either.

**With authentication disabled, keyed idempotency uses one shared anonymous namespace, so two anonymous callers using the same key and the same request deduplicate against each other. That is the honest consequence of a deployment that deliberately removes caller isolation, and it is recorded here rather than hidden: with no trusted principal there is nothing to separate callers by, and inventing isolation from an IP, a claimed tenant or metadata would violate the identity rules above.

The store is in memory, and therefore single-instance.** The records live in a bounded in-process map, the oldest dropped once the store is full, which is sufficient for a single-instance deployment. They are lost when the process stops, so a retry after a restart runs again rather than being answered from a record that no longer exists. Durable or distributed idempotency requires configured persistence, which the gateway does not presume.

**Concurrent identical requests are coalesced, not just sequential repeats.** The first caller claims the key and every later caller with the same fingerprint waits on that claim's outcome instead of starting a second invocation. Without this, two simultaneous retries — the situation idempotency exists to handle — both call the provider. Cancellation belongs to the caller who owns it: a waiter going away does not cancel the invocation it is waiting on, and an owner failing gives every waiter that failure rather than starting another provider attempt inside the same request, before releasing the claim so a later caller may retry.

The record is consulted after the model is resolved and before any provider is called, so an unknown model is a model-not-found error even when the key has a cached record. A request without a key is never recorded and reports `not_applicable` (§5.2).

`indeterminate` is the one status the gateway cannot produce today: it is the honest answer to a repeat after a timeout against a provider with no idempotency support, where the gateway cannot say whether the upstream executed the request, and it is reserved for that case. Because a failure is not recorded, the gateway currently raises the failure and a later retry executes again rather than reporting uncertainty — so the status names a state the implementation does not surface, and it is recorded here so a durable store knows what it means.

Idempotency applies to the non-streaming invocation only, deliberately. A replayed stream would be a synthetic stream of an answer that was never produced token by token in this call, which contradicts the standing rule that the gateway never presents a non-streamed answer as streaming (§5.3, §16): a request carrying a key is served as a stream like any other, and its status stays `not_applicable`.

---

# 13. Usage and Cost Accounting

Record:

- Invocation ID.
- Timestamp.
- Route and public model ID.
- Actual upstream model.
- Capability.
- Input tokens.
- Output tokens.
- Cached tokens when reported.
- Provider latency.
- Request status.
- Retry/fallback count.
- Estimated cost.

Token counts must distinguish provider-reported values from estimates.

Cost estimation must use configured pricing data, not hardcoded vendor-name assumptions.

Example:

```dotenv
USAGE_TRACKING_ENABLED=true
COST_TRACKING_ENABLED=true
USAGE_STORAGE=memory
```

Supported storage modes may include:

- `none`
- `memory`
- `sqlite`
- `postgres`

The default standalone deployment must not require an external database.

**The mechanism is implemented.** Every invocation that produces an answer leaves a record naming the invocation identifier and timestamp, the route that served it, the public model it was asked for, the upstream model that answered, the route's advertised capabilities, the provider's latency and attempt count, and whether a fallback was used. All three operations record, and the record is built in the inference service, which is the only layer that knows which route actually served the answer, so the route facts are not reconstructed by a caller of the service and a fallback answer is attributed to the route that produced it rather than to the plan's primary. The attempt count and the fallback flag reach the record from the walk rather than from the answer, because the canonical embeddings and image answers carry no field for either (§5.2) while the record has one for both. A tenant is recorded only where the request stated one: the canonical embeddings and image requests have no tenant field, because the tenant an invocation acts under is the credential's (§5.1), so those records state none rather than borrowing the namespace's value for a field the request never carried.

A token count reaches a record as the provider reported it or as the gateway derived it, and the difference travels with the number: `estimated_tokens` names the fields the gateway computed rather than read, so a derived count is never a reported one. Rendering an unreported count as zero — which the HTTP and gRPC usage shapes must do, because they type their counters as integers — turns an unknown quantity into an asserted one, so the wire layers make that trade-off and state it; the record does not.

Cost is estimated only when the deployment configures prices, from `COST_PRICING_JSON` keyed by public model ID, with prices per million tokens — the unit providers quote. A deployment that configures none records usage and no cost, which is the honest state of a deployment that has not said what its models cost. A malformed pricing value is refused at startup as a configuration error rather than partially applied: a deployment that misconfigures its prices should learn now, not bill wrong amounts for a month. A keyed but unpriced model is not silently zero-priced either — it is simply not priced.

Storage is in memory by default, which is sufficient for a single-instance deployment and requires no external database. Records are held in a bounded list, the oldest dropped once the store is full, so a deployment that never reads them cannot grow without limit. The bound weakens the replay guarantee in a stated way: a repeat is deduplicated only while its record remains available in this instance. A deployment that needs the guarantee across restarts or instances configures durable storage, which is a separate concern requiring configured persistence. The durable modes are named by the settings vocabulary but not implemented here.

Recording is not a part of serving the request: a sink that fails must not fail the invocation, because the answer has already been produced and the caller is waiting on it, but it must not fail the invocation silently either. The failure is logged for the operator instead, so a broken store cannot quietly turn into a month of unaccounted usage. Recording is also deliberately not part of idempotency's own record: the idempotency store holds the answer, and conflating the two would make an invocation's accounting depend on whether its caller happened to deduplicate.

---

# 14. Security

The gateway must implement:

- TLS support.
- Inbound authentication.
- Secret redaction.
- Request-body size limits.
- Provider timeouts.
- SSRF screening of configured upstream URLs and remote image URLs, backed by the transport-level controls in §14.1.
- Redirect restrictions; automatic redirect following is disabled at the transport, so a redirect is never followed automatically and surfaces as an error rather than a silent hop. Controlled redirect handling — a same-origin hop permitted after revalidation, a cross-origin hop only by explicit opt-in — is specified in §14.1 and not yet implemented.
- Host and scheme validation.
- Optional upstream hostname allowlists.
- Safe error normalization.
- Tenant-aware authorization where enabled.
- Per-client rate limits.
- Per-tenant concurrency limits.

Development may permit plain HTTP and unauthenticated local operation when explicitly configured.

Production must not silently fall back to insecure behavior.

## 14.1 SSRF protection and transport-level controls

Configuration-time screening alone does not constitute SSRF prevention. The gateway validates an upstream URL when configuration is loaded: scheme, blocked address classes, and every address the hostname resolves to. But the request is performed later by the inference binding, which resolves the hostname again and may follow redirects automatically. Validation and connection therefore live in different layers, leaving a DNS-rebinding (time-of-check to time-of-use) gap: a hostname can resolve to a public address during validation and to `127.0.0.1` or `169.254.169.254` at connection time.

Whatever transport performs the outbound request must therefore supply the controls that close that gap:

- **Address pinning (DNS pinning).** The connection must be pinned to an address that passed the gateway's policy, so the socket terminates at a validated destination rather than at whatever the hostname resolves to at connect time.
- **No automatic redirect following.** Automatic redirect following must be disabled, so that every redirect is evaluated by the gateway's redirect policy instead. A same-origin redirect may proceed once its destination is revalidated under the same rules. A cross-origin redirect must be rejected by default and permitted only through an explicit per-route opt-in, because an inference request is a POST that may carry custom authentication headers, provider or account headers, and a body containing private prompts and tool payloads; deciding which of those pieces may cross an origin is riskier than refusing the hop. Credentials are never forwarded across an origin change (§7).

Both obligations are executable in the binding today, not aspirational requirements on a future transport. The binding injects a client into every upstream call — the inference library honours a per-call `client=` argument — and that client supplies both controls:

- **The connection is pinned to a validated address.** The client resolves the hostname again, but accepts an address only after the gateway's policy has validated every answer, and connects to a validated one; the original hostname is preserved for TLS SNI and the `Host` header, so the provider still sees the host it expects. The verdict and the destination come from the same resolution, which closes the time-of-check to time-of-use gap above: the socket terminates where policy was checked, not wherever the hostname points at connect time.
- **Redirect following is disabled.** The client is built with redirect following off, because the inference library hardcodes it on for the client it would otherwise use and exposes no parameter that turns it off; a redirect therefore surfaces instead of being followed silently. (The per-call `transport=` argument is not an alternative: the library does not honour it as a transport and passes it to the request as a body parameter.)
- **Every client the library builds on that call's path is pinned, including one it rebuilds itself.** Which object `client=` must carry depends on the HTTP path the library takes for the route's dialect, and one of those paths answers a connection error by rebuilding its client from the handler's own transport attribute. The binding hands that path the pinned transport itself rather than a client wrapped around one, so the rebuilt client is pinned and non-redirecting too and the internal retry cannot leave the controls.

The injected client is not optional. Without it the call runs on the library's default client, which re-resolves the hostname at connect time and follows redirects — both controls are lost silently, and the configuration-time screening above is all that remains, which this section already states is not SSRF prevention.

**Where the boundary sits today.** A dialect is run only when the binding has verified that the library's HTTP path for it honours the injected client, because a path that discards that client substitutes the library's own default client with nothing raised. The verified set today is `openai`, which takes the OpenAI SDK path, and `anthropic` and `openrouter`, which share the generic HTTP path. A dialect outside that set is refused — at startup, so the deployment does not serve behind controls that would silently not apply, and again at the attempt — rather than run with the controls lost. Such a dialect is therefore not merely unsupported: adding one means verifying its HTTP path and its client handling, not only naming it in configuration.

**Redirects are refused, not evaluated.** Disabling redirect following is what makes a redirect visible; it does not decide the hop. The binding has no redirect-evaluation step today: a 3xx is neither followed nor assessed, and it surfaces to the caller as an upstream error — verified against the pinned library, an internal error on the OpenAI SDK path and a connection failure on the generic path — while the redirect destination is never contacted. The URL policy's redirect decision is therefore not in force: a same-origin hop permitted once its destination is revalidated under the same rules, a cross-origin hop refused unless a route opts in explicitly, and credentials never carried across an origin change are specified as policy and exercised by their own tests, but no request path calls that decision. Controlled redirect handling is not yet implemented. Until a caller evaluates the hop and re-issues the request, refusing every redirect is the safe half of the requirement, not the whole of it.

These obligations fall on the binding and its transport, not on the configuration loader. The two guarantees are not interchangeable:

- **Transport pinning** binds the request to the exact address the application validated, so the socket terminates where policy was checked.
- **Network enforcement** prevents reaching the prohibited address classes regardless of how DNS changes between validation and connection.

**Is a network-level egress restriction required?** Not for the dialects the binding runs. Pinning and disabled redirect following are enforced on every outbound connection those paths make, including the client the generic path's connection-error retry rebuilds, so the transport controls are complete for them and a deployment-level egress restriction is a defence-in-depth recommendation rather than a precondition. It remains the required compensating control wherever application-level address pinning cannot be enforced — a dialect or an operation the binding has not verified — and because such a dialect is refused rather than run, that requirement is met by the refusal: deployment-level egress controls must independently prevent access to every address class the gateway policy prohibits — loopback, private, link-local and metadata, unique-local, and other non-global destinations — for any path the binding does not verify and the deployment enables anyway. Any proxy used as that control must enforce equivalent destination validation itself. This is a compensating control, not literal address pinning: a firewall does not share the gateway's validated DNS answer, and a proxy that re-resolves the hostname independently recreates the same time-of-check to time-of-use gap.

A deployment relying on that compensating control must document it; it must not rely on configuration-time screening alone.

The controls must be proven by tests that exercise the rebinding and redirect cases against the transport actually used, not by unit tests of the URL validator alone. The rebinding case is proved that way today: the transport's own tests drive it through a stub inner transport and assert the address the request was actually sent to. The redirect case is proved as the refusal it currently is, and a redirect-evaluation path, once it exists, is what those tests must then exercise.

## 14.2 Inbound authentication scope

When authentication is enabled, all endpoints require a valid credential, except the health endpoints (`/healthz`, `/readyz`; §4.1, §4.2) when `GATEWAY_ALLOW_ANONYMOUS_HEALTH` is true. A presented credential is validated even on an exempt path, so a bad credential is never waved through: exemption waives the requirement to present a credential, not the validation of one that is presented. When authentication is disabled, every request proceeds with the anonymous identity (§7).

Caller-supplied `actor_id` and `agent_id` are attribution labels: they record who or what the caller says it is, and they must never become principals for quotas or authorization. Only the credential's tenant and its internal credential identifier may serve as principals (§5.1, §7).

---

# 15. Reliability

Recommended environment variables:

```dotenv
PROVIDER_TIMEOUT_SECONDS=110
PROVIDER_CONNECT_TIMEOUT_SECONDS=10
PROVIDER_RETRY_COUNT=2
PROVIDER_RETRY_BACKOFF_MS=500
PROVIDER_MAX_CONCURRENT_REQUESTS=100
GATEWAY_SHUTDOWN_TIMEOUT_SECONDS=30
```

The gateway must:

- Propagate cancellation.
- Enforce request deadlines.
- Bound retry duration.
- Respect upstream rate limits.
- Shut down gracefully.
- Avoid leaking connections.
- Prevent unbounded request queues.
- Return deterministic error categories.

A retry must not exceed the caller's remaining deadline.

The gateway enforces a single total deadline per invocation. Provider timeouts, connect timeouts, retries, and fallbacks all draw from that one budget. LiteLLM's retry and fallback support is disabled so that no second loop and no second deadline exists (§11.3).

**One deadline covers the whole invocation.** A single deadline is anchored when the invocation begins, and every attempt — the first attempt of the primary route, every retry, and every fallback — derives its budget from it, so a retry or a fallback consumes what is left rather than starting a fresh budget. Time is subtracted, never reset: the clock is anchored once, and each attempt is handed the smaller of the configured provider timeout and what the deadline then has left. **An attempt whose budget is exhausted is never started** — the provider is not called and a deadline error is raised instead (§11.2).

**The retry backoff is jittered, and a backoff that would consume the remaining budget is refused rather than slept away.** The delay before a retry is the exponential value extended by a uniformly random fraction of itself, so callers that failed together during a shared provider outage do not wake in a synchronised wave and hit the recovering provider at the same instant; the exponential value is the floor, so jitter only ever extends a wait and never makes a retry fire earlier than the configured backoff intends. The randomness is injectable — the schedule is a function of an injected random source, defaulting to the module's own — so a test passes a fixed source and reproduces the schedule exactly. A delay that would consume the invocation's whole remaining budget fails the deadline: it is refused with a deadline error rather than clamped to what is left and slept away, because sleeping until the deadline expired would spend the budget on a retry that could never then run, so the gateway reports the deadline immediately and leaves the budget unconsumed. In the retry loop that refusal ends the route's retries — the invocation falls back when another route is eligible, and otherwise reports the attempt's own failure (§11.2) — rather than waiting out a retry that could not be made.

**Simultaneous upstream attempts are bounded by the configured maximum, and so is the queue waiting for one.** `PROVIDER_MAX_CONCURRENT_REQUESTS` caps the number of attempts in flight, and the number of callers queued for a slot is bounded separately — by a multiple of that maximum — so a caller that arrives when the queue is full is rejected **immediately** with a resource-exhausted error (`CONCURRENCY_LIMIT_EXCEEDED`) rather than being queued. The bound is what keeps an overload from turning into unbounded resident work: a queue without a limit would hold every waiting caller's payload and framework state in process memory while most of them waited for a turn that would never come, and a caller refused for a full queue is told to back off now — the answer it would have reached after a long wait, minus the memory and the latency. No admission order is promised: queued waiters are admitted as slots free, and the limiter relies on no particular order — a caller that needs a strict arrival-order guarantee needs an explicit queue, which this limiter deliberately is not.

**The wait for a slot is itself bounded by the invocation deadline.** The limiter is handed what is left of the invocation's deadline, and a wait that outlives it raises a deadline error instead of blocking until a slot happens to free. An invocation that cannot be admitted before its deadline therefore fails promptly rather than staying queued, and under a saturated provider it does not stay resident holding its payload while ahead of work that could actually be served.

**A concurrency slot is held only for the duration of an attempt and never across a backoff wait**, so a caller sleeping between retries holds nothing another caller could be using, and a few slow callers cannot starve the rest. Waiting for a slot consumes invocation budget too, so an attempt's budget is computed at the moment the attempt starts, after the slot is held. A slot is released on every exit from an attempt — success, failure, or cancellation — so a cancelled attempt frees its slot for the next caller, and a caller whose wait ran out never acquires one.

**Cancellation propagates.** A cancelled invocation stops at the attempt in flight, releases the slot that attempt held, and makes no further attempt. Cancellation is a :exc:`BaseException` the reliability loop never catches, so it is never converted into an upstream failure, never classified as retryable, and never retried into.

**`attempt_count` counts gateway attempts, never upstream requests.** The response's attempt count is the number of calls the gateway made — one per attempt, on one route, under one attempt number. The inference library has internal retry paths that no supported knob disables (§11.3), so one gateway attempt can carry more than one wire request and this count may understate what the upstream saw. It must never be read, reported, or recorded as a wire-request count.

---

# 16. Streaming

HTTP streaming must use Server-Sent Events where the endpoint's protocol defines SSE.

The guarantees below are the HTTP endpoint's. The gRPC streaming of §5 is not yet implemented — this codebase runs no gRPC server — so this section makes no claim about it.

Streaming is delivered through the LiteLLM binding — the same entry point as a non-streaming call with streaming switched on (§9.6). The gateway maintains no per-provider streaming adapters: it consumes the stream the binding produces for the configured provider and re-normalises each chunk at the gateway boundary. The gateway precedes the first canonical event with a conventional assistant-role delta — a subset-compatibility measure for OpenAI-compatible clients, not a canonical event — so the canonical stream is the five kinds below.

**The stream is a sequence of canonical events.** Every event is one of five kinds, and the five are the whole vocabulary. `content` carries incremental text. `tool_call` carries one tool-call delta: the index of the call being built, and any of an id, a name and a fragment of the arguments. `usage` carries a usage snapshot, where the upstream reports one. `finish` is the terminal event of a stream that completed: the reason generation stopped, classified in the gateway's own vocabulary with the provider's own raw reason beside it. `error` is the terminal event of a stream that failed after it had produced output: the gateway's own stable reason and its sanitised message, and no finish reason, because the provider did not finish. Every event carries the id of the invocation whose stream it belongs to, so a consumer that multiplexes streams attributes each one, and every event carries its own 0-based position in that invocation's stream — its sequence number. Within one invocation sequence numbers are strictly increasing from zero, so a consumer can tell that events arrived in the order the gateway produced them; a repeated, skipped or backwards sequence number is a broken stream. The numbers cannot by themselves prove that no trailing event was lost: a stream that stops before its last event shows no gap, because the events that were never sent were never numbered. The sequence restarts at zero with each new invocation.

**A stream ends in exactly one of three ways, and a consumer can tell which.** The three outcomes are exhaustive. A stream that **completed** ends with exactly one `finish` event, which is always its last event; the endpoint then emits `[DONE]`. A stream that **failed** after it had produced output ends with exactly one `error` event, which is always its last event; the endpoint closes the stream without `[DONE]`. A stream that was **truncated** ends with no terminal event at all — the client went away, the transport was lost, or the upstream ended without ever reporting a finish reason — and the endpoint closes it without `[DONE]` either. `[DONE]` is therefore emitted only after a `finish` event and never after an `error` event or a truncation, so the terminator's presence or absence distinguishes all three outcomes. Nothing may follow a terminal event: the gateway stops the stream the moment it yields a `finish` or an `error`, so a stream produces at most one, and a failure that surfaces after a terminal event has been yielded is reported as the invocation's own error rather than as a second terminal event. A refusal before the first event — an unknown model, a capability gap, a deadline, a provider that cannot stream — ends the invocation with an ordinary error response rather than a stream, and produces no event at all.

**A failure after output is an error event, not an unknown finish reason.** The terminal `error` event carries the gateway's own stable reason — the machine-readable value a caller switches on — and the same sanitised message the gateway reports for the invocation, so the reason and the text agree with the error that is raised alongside the event. It carries no finish reason. A finish reason classifies why generation stopped, and on this path generation did not stop but failed, so presenting the failure as a finish — borrowing the `unknown` reason and parking the gateway's reason in the raw-reason field — would put gateway vocabulary in a field reserved for the provider's own, and would render a failure as a finish reason to every consumer of the OpenAI-compatible shape. An upstream stream that ends without ever reporting a finish reason is classified deliberately rather than presented as a successful completion: the gateway synthesises no terminal event for an answer it never saw finish, so that stream ends with no terminal event and reads as truncated, never as completed. On a successful stream the raw reason beside the classification is the provider's own reason whenever the inference library kept it: the binding reads the library's native-reason field when the library retained one, exactly as the non-streaming path does, so an unrecognised provider reason arrives verbatim and is classified `unknown` with the raw value preserved. Only a reason the library normalised without retaining is lost: the library maps an unrecognised provider reason to `stop` before the gateway boundary, and that wording cannot be reconstructed downstream — the gateway records the value it was handed rather than inventing a mapping to pretend otherwise.

**A provider that cannot stream is refused, never faked.** Streaming is a capability of the provider the gateway is bound to, not a capability a route advertises: `streaming` is absent from the route capability vocabulary (§6), and no route check tests for it. The gateway resolves the model first — a request naming a model the gateway does not expose is refused as a model-not-found error whatever the bound provider can do — and then, before any provider is called, asks whether the bound provider implements the streaming protocol; when it does not, the request is refused with a capability error naming `streaming`, so the refusal reaches the caller as an ordinary error response rather than a stream. The gateway never answers a streaming request with a complete, non-streamed body, because presenting a whole answer as it arrived would tell the caller it received tokens as they were produced, which it did not.

**A stream is bounded as a non-streaming invocation is.** The invocation's single deadline covers the whole stream: it is anchored when the invocation begins, every attempt draws its budget from it, and an attempt that cannot fit inside what remains is never started (§15). A stream holds a concurrency slot for its duration and releases it on every exit; a caller waiting past the configured bound is refused. Cancellation propagates: a cancelled stream stops at the attempt in flight, releases the slot that attempt held, and is never converted into an upstream failure.

**A client disconnect stops the upstream work.** When the caller goes away mid-stream, the gateway stops the in-flight upstream attempt rather than spending a provider's capacity on output nobody is left to receive, and releases the slot that attempt held. No terminal event is emitted on this path — the stream simply ends — so a consumer that loses its connection reads a truncated stream: one that stopped without a terminal event and without `[DONE]`, not a failed one.

---

# 17. Observability

Required metrics:

```text
gateway_requests_total
gateway_request_duration_seconds
gateway_upstream_requests_total
gateway_upstream_duration_seconds
gateway_tokens_input_total
gateway_tokens_output_total
gateway_retries_total
gateway_fallbacks_total
gateway_errors_total
gateway_active_requests
gateway_rate_limit_rejections_total
```

Distributed tracing should support W3C Trace Context and OpenTelemetry.

Logs must be structured.

Prompts, responses, credentials, and private tool arguments must not be logged by default.

---

# 18. Complete Environment Example

```dotenv
# ======================================
# SERVER
# ======================================
CHORA_HTTP_PORT=8080
CHORA_GRPC_PORT=9090
CHORA_ENVIRONMENT=production

# ======================================
# TEXT GENERATION
# ======================================
TEXT_LLM_ENDPOINT=https://api.example.com/v1
TEXT_LLM_MODEL_ID=text-model
TEXT_LLM_API_KEY=
TEXT_LLM_FORMAT=openai
TEXT_LLM_AUTH_TYPE=bearer
TEXT_LLM_PATH=
TEXT_LLM_EXTRA_HEADERS_JSON={}
TEXT_LLM_CAPABILITIES=chat,tools,vision,structured-output
TEXT_LLM_CONTEXT_WINDOW=
TEXT_LLM_MAX_OUTPUT_TOKENS=
TEXT_LLM_ROUTES_JSON=[]

# ======================================
# IMAGE GENERATION
# ======================================
IMAGE_GEN_ENDPOINT=https://images.example.com/v1
IMAGE_GEN_MODEL_ID=image-model
IMAGE_GEN_API_KEY=
IMAGE_GEN_FORMAT=openai
IMAGE_GEN_AUTH_TYPE=bearer
IMAGE_GEN_PATH=
IMAGE_GEN_EXTRA_HEADERS_JSON={}
IMAGE_GEN_CAPABILITIES=generation
IMAGE_GEN_ROUTES_JSON=[]

# ======================================
# EMBEDDINGS
# ======================================
EMBEDDINGS_ENDPOINT=https://embeddings.example.com/v1
EMBEDDINGS_MODEL_ID=embedding-model
EMBEDDINGS_API_KEY=
EMBEDDINGS_FORMAT=openai
EMBEDDINGS_AUTH_TYPE=bearer
EMBEDDINGS_PATH=
EMBEDDINGS_DIMENSIONS=1024
EMBEDDINGS_SUPPORTS_DIMENSIONS_PARAM=false
EMBEDDINGS_CAPABILITIES=embeddings
EMBEDDINGS_ROUTES_JSON=[]

# ======================================
# AUTHENTICATION
# ======================================
GATEWAY_AUTH_ENABLED=true
GATEWAY_API_KEYS=primary-key,internal-tooling:tenant-acme

# ======================================
# RELIABILITY
# ======================================
PROVIDER_TIMEOUT_SECONDS=110
PROVIDER_CONNECT_TIMEOUT_SECONDS=10
PROVIDER_RETRY_COUNT=2
PROVIDER_RETRY_BACKOFF_MS=500
PROVIDER_MAX_CONCURRENT_REQUESTS=100

# ======================================
# USAGE
# ======================================
USAGE_TRACKING_ENABLED=true
COST_TRACKING_ENABLED=false
USAGE_STORAGE=memory

# ======================================
# OBSERVABILITY
# ======================================
OTEL_ENABLED=false
OTEL_EXPORTER_OTLP_ENDPOINT=
LOG_LEVEL=info
```

---

# 19. Deployment

The gateway must support running as a single standalone process.

Container deployment must expose:

```text
8080/tcp  HTTP
9090/tcp  gRPC
```

The process must:

1. Load environment configuration.
2. Validate enabled capabilities.
3. Initialize the LiteLLM binding.
4. Initialize optional persistence.
5. Initialize observability.
6. Start HTTP and gRPC listeners.
7. Publish readiness.
8. Handle shutdown signals gracefully.

The service must not require Kubernetes, a service mesh, or a cloud-managed runtime.

---

# 20. Testing Requirements

## Unit tests

Cover:

- Configuration parsing.
- Environment validation.
- URL resolution.
- Authentication headers.
- Request normalization.
- Response normalization.
- Public model ID resolution.
- Retry classification.
- Error mapping.
- Tool-call translation.
- Embedding dimensions.

## Integration tests

Use deterministic mock upstream servers that implement the provider protocol the binding speaks for each configured format. The mocks stand in for the upstream provider, never for the gateway: the gateway itself maintains no per-provider protocol implementation.

The public gateway contract must stay covered. Mock providers substitute for the upstream only, and the gateway's own HTTP and gRPC surface is exercised against them.

Test:

- OpenAI.
- Anthropic.
- Vertex.
- Gemini.
- Ollama.
- Images.
- Embeddings.
- Streaming.
- Provider failures.
- Cancellation.
- Authentication.
- Timeouts.
- Fallbacks.
- URL policy enforcement: rebinding and redirect handling exercised against the transport actually used (§14.1).

## End-to-end tests

Start the gateway with environment variables and mock providers.

Verify every HTTP and gRPC endpoint without requiring a real commercial provider account.

At least one optional live-provider smoke test should demonstrate switching providers through environment configuration alone.

---

# 21. Acceptance Criteria

The product is accepted when:

1. The gateway starts independently without external application services.
2. Text inference works through at least four configured LiteLLM providers.
3. Text providers can be switched without recompilation.
4. Image generation uses independently configured credentials and endpoints.
5. Embeddings use independently configured credentials and endpoints.
6. HTTP and gRPC share canonical application logic.
7. Tool calls round-trip without losing arguments or identifiers.
8. Multimodal inputs preserve supported content.
9. Unsupported capabilities produce explicit errors.
10. Streaming works for compatible providers.
11. Timeouts, retries, and fallbacks behave deterministically.
12. No provider credentials are exposed through public APIs.
13. Unknown public model IDs cannot reach unconfigured upstream endpoints.
14. Usage accounting distinguishes actual and estimated token counts.
15. The service can run without a database.
16. Health and readiness checks function independently.
17. Configuration errors are detected at startup.
18. Automated tests cover all public endpoints.
19. The implementation contains no vendor-specific business logic outside the LiteLLM binding.

---

# 22. Implementation Milestones

**Phase 1 — Core gateway**

- Configuration loader.
- HTTP and gRPC servers.
- Canonical request/response types.
- Model routing.
- LiteLLM binding for text inference.
- Authentication and health checks.

**Phase 2 — Multi-provider support**

- Additional LiteLLM providers (Anthropic, Vertex, Gemini, Ollama, Azure).
- Tool calling.
- Vision support.

**Phase 3 — Additional capabilities**

- Image generation.
- Image editing.
- Embeddings.

**Phase 4 — Production hardening**

- Streaming.
- Retries and fallbacks.
- Rate limiting.
- Concurrency limits.
- Usage and cost accounting.
- Metrics and tracing.
- Security testing.

**Phase 5 — Conformance and release**

- Protocol conformance suites.
- Mock-provider integration tests.
- Deployment documentation.
- Example environment files.
- Container image.
- Versioned API and protobuf contracts.

---

# 23. Final Architectural Requirements

The gateway must maintain three independent layers.

**Northbound interfaces**

Stable HTTP and gRPC contracts consumed by applications.

**Gateway core**

Vendor-independent routing, validation, governance, reliability, and accounting.

**LiteLLM binding**

Protocol-specific request encoding, network execution, and response decoding, delegated to the LiteLLM library.

Vendor neutrality is a property of the gateway, not of its inference dependency. The gateway core holds no provider-specific business logic, and the provider is chosen entirely by configuration. That neutrality is bounded by the LiteLLM dependency: the gateway is vendor-neutral to the extent that LiteLLM can speak the target provider's protocol. The dependency is pinned, isolated behind the binding, and replaceable — swapping it changes the binding, not the gateway's public contract, routing, policy, or accounting.

No commercial vendor identity may be required by the gateway core.

A provider change must normally require only environment configuration changes.

The gateway must be usable as an independent infrastructure component, regardless of which application, programming language, deployment platform, or AI provider is connected to it.