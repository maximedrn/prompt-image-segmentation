---
name: python-to-effect-style
description: >
  Refactor an existing conventional Python codebase into an Effect-TS-like architecture:
  typed recoverable failures, explicit capabilities/dependencies, composable effects,
  deterministic resource lifetimes, structured concurrency, retry/timeout policies,
  validated boundaries, and testable pure business logic. Use when modernizing Python
  services that rely on broad exceptions, hidden globals, ad-hoc dependency injection,
  unmanaged asyncio tasks, ambiguous None values, mutable shared state, or untyped dicts.
---

# Python → Effect-Style Refactoring Skill

## Mission

Refactor **existing ordinary Python** into a disciplined architecture inspired by Effect-TS.

The source is assumed to look like normal Python:

- functions that raise exceptions;
- classes that import global clients;
- `None` used for failure and absence interchangeably;
- `try/except Exception`;
- `asyncio.create_task`;
- network/database calls mixed with business rules;
- mutable service objects;
- framework dependency injection leaking into domain code;
- `dict[str, Any]`;
- resources opened and closed manually;
- retry loops written inline;
- tests built around monkeypatching.

The destination should behave like an Effect-oriented application:

```text
Effect-like operation
    =
explicit requirements
+ explicit recoverable errors
+ explicit success type
+ controlled side effects
+ owned resource lifetime
+ owned concurrency
```

When using `stateless`, the conceptual target is:

```python
Effect[Requirements, Error, Success]
```

This corresponds mentally to Effect-TS:

```ts
Effect<Success, Error, Requirements>
```

The goal is **not** to rewrite Python so it visually resembles TypeScript.

The goal is to make the important properties of the program visible and composable.

---

# 1. Default target architecture

For a Python codebase that should become genuinely Effect-like, prefer:

```text
Python 3.12+
│
├─ strict typing
│
├─ stateless
│   ├─ Effect[Requirements, Error, Success]
│   ├─ Need[T]
│   ├─ typed recoverable errors
│   ├─ handlers
│   ├─ async/wait
│   └─ retry/schedule where appropriate
│
├─ AnyIO
│   ├─ structured concurrency
│   ├─ cancellation
│   ├─ timeout
│   └─ synchronization
│
├─ contextlib
│   ├─ contextmanager
│   ├─ asynccontextmanager
│   └─ AsyncExitStack
│
├─ Pydantic
│   └─ validation at external boundaries
│
└─ optional Dishka
    └─ only for complex scoped dependency graphs
```

`stateless` is the default effect abstraction for this skill because it models:

```python
Effect[Abilities, Error, Result]
```

and supports explicit abilities/dependencies through `Need[T]`.

Do not add `returns` as a second universal abstraction once `stateless` is the chosen core.

---

# 2. When NOT to use `stateless`

Use a conservative Python-native version of the same architecture when:

- the team will not maintain generator-based effect code;
- strict type checking cannot be enforced;
- the codebase is small enough that explicit dependency arguments remain clearer;
- library-risk policy forbids adopting a younger effect library;
- framework constraints make the effect runtime awkward.

The fallback architecture is:

```text
Protocol
+ immutable domain types
+ explicit typed domain exceptions
+ explicit function/constructor dependencies
+ AnyIO
+ context managers
+ Pydantic
```

The fallback must preserve the same invariants.

It is not permission to return to globals and `except Exception`.

---

# 3. Supporting alternatives

## `returns`

Use `returns` if the desired architecture is primarily:

- `Result`;
- `Maybe`;
- Reader/dependency passing;
- async result composition.

Useful types include:

```python
Result[T, E]
Maybe[T]
FutureResult[T, E]
RequiresContext[T, Env]
RequiresContextResult[T, E, Env]
RequiresContextFutureResult[T, E, Env]
```

It is a good functional toolkit, but this skill does not treat it as the default full Effect-like runtime.

## `aleff`

Investigate `aleff` only for specialized algebraic-effect requirements such as:

- multi-shot continuations;
- backtracking;
- nondeterminism;
- deep/shallow handlers;
- resumable handlers that must resume more than once.

Do not select it by default for ordinary service refactoring.

---

# 4. Core transformation rule

For every non-trivial Python operation, identify three things:

```text
What does it need?
How can it fail recoverably?
What does it return on success?
```

Convert implicit behavior into a visible contract.

Ordinary Python:

```python
async def create_user(email: str) -> User:
    ...
```

may secretly mean:

```text
needs:
- database
- payment/customer API
- logger
- clock

recoverable failures:
- EmailAlreadyExists
- DatabaseUnavailable
- CustomerProviderUnavailable

success:
- User
```

The Effect-like target should make these properties discoverable from the operation and its dependencies.

Conceptually:

```python
Effect[
    Need[UserRepo] | Need[CustomerGateway] | Need[Clock],
    EmailAlreadyExists | DatabaseUnavailable | CustomerProviderUnavailable,
    User,
]
```

Do not add dependencies or errors to the type merely because they exist somewhere lower in the stack.
Model the contract actually relevant to the operation.

---

# 5. Mandatory refactoring sequence

Do not convert the whole codebase to effects in one pass.

Use the following order.

## Phase 0 — inventory the current Python bullshit

Search the repository for:

```text
except Exception
except BaseException
raise Exception
raise ValueError
return None
Optional[
| None
asyncio.create_task
asyncio.gather
global
requests.
httpx.
aiohttp.
open(
Session(
Client(
os.environ
dict[str, Any]
typing.Any
monkeypatch
patch(
sleep(
retry
while True
```

Also inspect:

- singleton modules;
- dependency containers;
- framework `Depends`;
- database session creation;
- HTTP client creation;
- background workers;
- job runners;
- CLI entrypoints.

Classify every finding into:

```text
typed failure
defect
hidden dependency
resource lifetime
concurrency
validation boundary
mutable shared state
untyped data
retry/timeout
```

Do not refactor blindly before this inventory.

---

## Phase 1 — create domain types

Move away from primitive soup.

Bad:

```python
async def charge(
    user_id: str,
    amount: float,
    currency: str,
) -> dict[str, Any]:
    ...
```

Better:

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class UserId:
    value: str

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

@dataclass(frozen=True)
class Receipt:
    id: str
    amount: Money
```

Do this before introducing sophisticated effect composition.

A typed effect returning `dict[str, Any]` is still bad architecture.

---

## Phase 2 — build an error taxonomy

List all currently raised exceptions.

For each one, decide whether it is:

```text
recoverable domain failure
recoverable infrastructure failure
validation failure
cancellation/control flow
defect/programmer error
```

Then create explicit error types for recoverable failures.

Example:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class EmailAlreadyExists(Exception):
    email: str

@dataclass(frozen=True)
class DatabaseUnavailable(Exception):
    operation: str

@dataclass(frozen=True)
class PaymentDeclined(Exception):
    reason: str
```

Do not create:

```python
class AppError(Exception):
    pass
```

and map everything to it.

The error taxonomy should become **more precise**, not less.

---

## Phase 3 — extract capabilities

Find code like:

```python
from app.database import db
from app.stripe import stripe
from app.settings import settings

async def create_user(...):
    ...
```

Replace hidden operational dependencies with capability interfaces.

Prefer `Protocol`:

```python
from typing import Protocol

class UserRepo(Protocol):
    async def find_by_email(self, email: str) -> "User | None":
        ...

    async def insert(self, user: "User") -> None:
        ...

class CustomerGateway(Protocol):
    async def create_customer(self, email: str) -> "Customer":
        ...
```

Capabilities describe **what application code needs**, not every method exposed by an SDK.

Bad:

```python
class StripeClientProtocol(Protocol):
    # 47 methods copied from SDK
```

Better:

```python
class CustomerGateway(Protocol):
    async def create_customer(self, email: str) -> Customer:
        ...
```

---

## Phase 4 — isolate infrastructure adapters

SDK/database exceptions must stop leaking upward.

Bad:

```python
async def create_user(...):
    try:
        await session.execute(...)
    except sqlalchemy.exc.IntegrityError:
        ...
```

if `create_user` is application/domain code.

Move infrastructure-specific handling into adapters:

```text
Postgres adapter
  DB exception
      ↓
DatabaseUnavailable / EmailAlreadyExists

Stripe adapter
  SDK exception
      ↓
PaymentProviderUnavailable / PaymentDeclined
```

Application logic should know application errors, not driver classes.

---

## Phase 5 — convert orchestration to effects

Once dependencies and errors are explicit, convert use cases.

Do not begin by wrapping every tiny helper in `Effect`.

Start with operations that:

- orchestrate multiple dependencies;
- may fail in several recoverable ways;
- involve retries/timeouts;
- need testable environmental capabilities;
- coordinate resources;
- coordinate concurrency.

Keep truly pure helpers as ordinary pure Python functions.

---

## Phase 6 — move resource ownership to bootstrap/scopes

Database engines, sessions, HTTP clients, file handles, transactions, temporary resources, and
subscriptions need explicit owners.

Use:

```python
with ...
```

```python
async with ...
```

```python
ExitStack
```

```python
AsyncExitStack
```

Never rely on garbage collection for resource correctness.

---

## Phase 7 — replace ad-hoc async with structured concurrency

Find all detached tasks.

Replace ownership-less concurrency with AnyIO task groups or an equivalent structured primitive.

Every task must have:

```text
owner
lifetime
cancellation behavior
failure propagation behavior
```

---

## Phase 8 — validate external inputs at boundaries

Move validation to:

- HTTP request parsing;
- config loading;
- message ingestion;
- CLI argument parsing;
- external API decoding.

Domain/application code should receive typed values.

---

## Phase 9 — migrate module by module

Do not perform a big-bang rewrite.

A module is migrated when:

- dependencies are explicit;
- domain failures are explicit;
- driver errors do not leak;
- resources are owned;
- async tasks are structured;
- boundaries are validated;
- tests do not require hidden-global monkeypatching.

---

# 6. `stateless` mental model

The important correspondence is:

```text
Effect-TS                     stateless

Effect<A, E, R>               Effect[R, E, A]

success A                     result A
recoverable error E           error E
requirements R                abilities / Need[T]
Context service               Need[Protocol]
provide service               supply(...)
effect handler                handler
async effect                  Async + wait
```

Do not mechanically port Effect-TS syntax.

Use Effect-TS only as the architectural model.

---

# 7. Basic `stateless` capability pattern

Example:

```python
from typing import Never
from stateless import Effect, Need, need

class Console:
    def print(self, line: str) -> None:
        print(line)

def greet(name: str) -> Effect[Need[Console], Never, None]:
    console = yield from need(Console)
    console.print(f"Hello {name}")
```

The key property is not the generator syntax.

The key property is that the operation declares:

```text
requires Console
cannot fail recoverably
returns None
```

---

# 8. Multiple dependencies

```python
from typing import Never
from stateless import Effect, Need, need

class Audit:
    def record(self, event: str) -> None:
        ...

class UserRepo:
    def get_name(self, user_id: str) -> str:
        ...

def load_and_audit(
    user_id: str,
) -> Effect[Need[UserRepo] | Need[Audit], Never, str]:
    repo = yield from need(UserRepo)
    audit = yield from need(Audit)

    name = repo.get_name(user_id)
    audit.record(f"loaded:{user_id}")
    return name
```

Requirements compose instead of becoming invisible imports.

---

# 9. Typed failures

Recoverable errors should be explicit.

Example:

```python
from dataclasses import dataclass
from stateless import Try, throw

@dataclass(frozen=True)
class UserNotFound(Exception):
    user_id: str

def require_user(
    user_id: str,
    existing: set[str],
) -> Try[UserNotFound, str]:
    if user_id not in existing:
        yield from throw(UserNotFound(user_id))

    return user_id
```

Follow the exact API/signature of the installed `stateless` version if `throw` invocation details differ.

Do not guess undocumented APIs during migration.

---

# 10. Distinguish failure from defect

This distinction is mandatory.

## Recoverable failure

The caller can reasonably choose another branch.

Examples:

```text
UserNotFound
EmailAlreadyExists
PaymentDeclined
RateLimited
DatabaseUnavailable
Conflict
Timeout
```

These belong in the typed error contract when relevant.

## Defect

Examples:

```text
assertion failed
impossible state
programmer bug
broken invariant
unexpected third-party contract violation
```

These should normally raise normally and remain visible.

Do not write:

```python
try:
    ...
except Exception as exc:
    yield from throw(AppError(str(exc)))
```

This turns defects into fake business errors.

---

# 11. Convert existing exceptions deliberately

A conventional adapter might be:

```python
async def load_user(user_id: str) -> User:
    try:
        return await db.fetch_user(user_id)
    except TimeoutError:
        raise DatabaseUnavailable("load_user")
```

During migration:

1. keep the external exception handling at the adapter boundary;
2. map only expected exceptions;
3. surface a typed application error;
4. preserve the original cause where useful.

Example:

```python
raise DatabaseUnavailable(operation="load_user") from exc
```

Then convert the adapter result/exception into the selected effect error channel at a narrow point.

Do not wrap every Python call with a broad effect-catching utility.

---

# 12. `None` migration rules

`None` is acceptable for actual absence.

Good:

```python
async def find_user(email: str) -> User | None:
    ...
```

when "not found" is naturally queried as optional data.

Bad:

```python
async def charge(...) -> Receipt | None:
    ...
```

if `None` means:

```text
declined
gateway timeout
invalid card
unknown customer
```

Those are different failures and must not be collapsed.

During migration, inspect every `return None`.

Ask:

```text
Does None mean absence,
or did somebody use None to avoid defining an error?
```

---

# 13. Kill broad exception handling

Search for:

```python
except Exception:
except Exception as exc:
```

Classify each catch.

Allowed uses are rare and typically at process boundaries:

```text
HTTP top-level exception middleware
worker/job runner boundary
CLI entrypoint
telemetry crash boundary
```

Even there, broad catches are for reporting/cleanup, not for pretending the operation succeeded.

Inside application logic, catch only errors you can meaningfully handle.

---

# 14. No global operational dependencies

Forbidden in migrated application/domain modules:

```python
db = Database(...)
client = httpx.AsyncClient(...)
stripe = Stripe(...)
settings = load_settings(...)
```

when imported elsewhere as ambient singletons.

Prefer:

```text
Protocol
→ concrete adapter
→ bootstrap construction
→ explicit supply
```

A global immutable constant is not the same as a hidden operational dependency.

This is fine:

```python
MAX_USERNAME_LENGTH = 80
```

This is not:

```python
global_database_session = ...
```

---

# 15. Framework DI is a boundary concern

FastAPI, Django, Flask extensions, task frameworks, or custom service containers may perform DI.

Do not let framework types become the domain architecture.

Bad:

```python
async def create_user(
    db: Session = Depends(get_db),
    stripe: Stripe = Depends(get_stripe),
):
    # core business logic
```

Better:

```text
FastAPI route
    ↓
resolve/request dependencies
    ↓
run application effect/use case
    ↓
map typed result to HTTP
```

Application code should be callable without booting the web framework.

---

# 16. Resource safety: make lifetime explicit

Find:

```python
client = ...
try:
    ...
finally:
    await client.close()
```

and especially code that forgets the `finally`.

Prefer:

```python
async with make_client() as client:
    ...
```

or:

```python
from contextlib import AsyncExitStack

async with AsyncExitStack() as stack:
    db = await stack.enter_async_context(open_database())
    http = await stack.enter_async_context(open_http_client())
    ...
```

Bootstrap owns long-lived resources.

Request/job scopes own short-lived resources.

A resource must not escape the scope that guarantees its finalizer unless ownership is explicitly transferred.

---

# 17. Structured concurrency

Search for:

```python
asyncio.create_task(...)
```

Every occurrence is suspicious until ownership is proven.

Bad:

```python
async def handle_request():
    asyncio.create_task(send_email())
    return {"ok": True}
```

Questions:

```text
What happens if send_email fails?
Who observes the exception?
Who cancels it at shutdown?
May it outlive the request?
May the process exit before it finishes?
```

Prefer structured concurrency:

```python
import anyio

async with anyio.create_task_group() as tg:
    tg.start_soon(task_a)
    tg.start_soon(task_b)
```

Use a real background-job system when work is intentionally detached from the request lifetime.

Do not disguise detached tasks as structured concurrency.

---

# 18. Timeout and cancellation

Timeouts must be part of operation semantics.

Prefer:

```python
import anyio

with anyio.fail_after(5):
    await operation()
```

using the exact API form required by the installed AnyIO version.

Do not scatter magic timeout numbers through adapters.

Prefer configuration/value objects:

```python
@dataclass(frozen=True)
class PaymentPolicy:
    timeout_seconds: float
```

Cancellation is control flow.

Do not accidentally swallow cancellation through broad exception handlers.

---

# 19. Retry as policy, not loop noise

Find patterns such as:

```python
for attempt in range(5):
    try:
        return await call()
    except Exception:
        await asyncio.sleep(1)
```

Replace them with explicit policy.

A retry policy must define:

```text
which errors are retryable
number of attempts
delay
backoff
jitter
maximum elapsed time
idempotency assumptions
```

Do not retry:

```text
validation errors
business rejection
programmer defects
non-idempotent operations without safeguards
```

Use `stateless` schedules/retry when they precisely match desired semantics.

Otherwise use a focused retry mechanism at the adapter/application boundary.

---

# 20. Pure logic stays pure

Do not make everything an Effect.

Bad:

```python
def add_tax(...) -> Effect[Never, Never, Money]:
    ...
```

when it is a deterministic calculation.

Prefer:

```python
def add_tax(amount: Money, rate: Decimal) -> Money:
    ...
```

Use effects for computations with meaningful:

```text
requirements
recoverable failures
environment interaction
resource/concurrency semantics
```

This prevents "Effect ceremony" from becoming a new form of bullshit.

---

# 21. Boundary validation

External data is hostile/untrusted until decoded.

Examples:

```text
HTTP JSON
environment variables
config files
Kafka messages
webhook payloads
third-party APIs
CLI input
database JSON blobs
```

Validate at the boundary.

Prefer Pydantic or an equivalent repository-standard validator.

Example:

```python
from pydantic import BaseModel, EmailStr

class CreateUserRequest(BaseModel):
    email: EmailStr
```

Then convert to domain values if the domain model differs.

Do not pass Pydantic/web framework models everywhere merely because they are validated.

---

# 22. Replace `dict[str, Any]`

Search for:

```python
dict[str, Any]
Mapping[str, Any]
Any
```

Classify each use.

At external boundaries, temporary untyped mappings may be unavoidable.

Inside business logic, replace them with:

```text
dataclass
TypedDict when shape-only structural typing is appropriate
Pydantic boundary model
Enum
Literal
NewType/value object
Protocol
```

Strict typing is part of the migration, not a cleanup task afterward.

---

# 23. Strict type checking

Prefer:

```text
pyright --strict
```

or the repository's existing strict checker.

Rules:

- do not add unexplained `Any`;
- do not use blanket `# type: ignore`;
- do not silence architecture problems with `cast`;
- annotate public functions;
- type capabilities with `Protocol`;
- type error unions deliberately.

If the library integration exposes checker limitations, isolate those limitations in a tiny compatibility module instead of spreading ignores across the repository.

---

# 24. Example: ordinary Python → Effect-like Python

## Before

```python
async def register(email: str):
    existing = await db.find_user(email)

    if existing:
        raise ValueError("already exists")

    try:
        customer = await stripe.create_customer(email)
        user = await db.insert_user(email, customer.id)
        asyncio.create_task(send_welcome_email(user))
        return user
    except Exception as exc:
        logger.exception(exc)
        return None
```

Problems:

```text
db is hidden
stripe is hidden
logger is hidden
mailer is hidden
ValueError is unstructured
all infrastructure failures collapse
defects collapse
None means failure
background task has no owner
no typed success/failure contract
```

## Step 1 — domain errors

```python
@dataclass(frozen=True)
class EmailAlreadyExists(Exception):
    email: str

@dataclass(frozen=True)
class CustomerProviderUnavailable(Exception):
    operation: str

@dataclass(frozen=True)
class DatabaseUnavailable(Exception):
    operation: str
```

## Step 2 — capabilities

```python
class UserRepo(Protocol):
    async def find_by_email(self, email: str) -> User | None:
        ...

    async def insert(
        self,
        email: str,
        customer_id: str,
    ) -> User:
        ...

class CustomerGateway(Protocol):
    async def create_customer(self, email: str) -> Customer:
        ...

class WelcomeMailer(Protocol):
    async def send_welcome(self, user: User) -> None:
        ...
```

## Step 3 — effect-like contract

Conceptually:

```python
Effect[
    Need[UserRepo] | Need[CustomerGateway] | Need[WelcomeMailer] | Async,
    EmailAlreadyExists | CustomerProviderUnavailable | DatabaseUnavailable,
    User,
]
```

## Step 4 — orchestration

The exact code must follow the selected `stateless` version, but the structure should become:

```python
def register(email: str) -> AppEffect[User]:
    repo = yield from need(UserRepo)
    customer_gateway = yield from need(CustomerGateway)
    mailer = yield from need(WelcomeMailer)

    existing = yield from wait(repo.find_by_email(email))

    if existing is not None:
        yield from throw(EmailAlreadyExists(email))

    customer = yield from create_customer(customer_gateway, email)
    user = yield from insert_user(repo, email, customer.id)

    # Deliberately decide whether email is:
    # - part of the transaction/use-case,
    # - structured concurrent work,
    # - or an external background job.
    yield from send_welcome(mailer, user)

    return user
```

The important improvement is architectural:

```text
requirements visible
recoverable failures visible
success visible
hidden globals removed
None ambiguity removed
background semantics decided explicitly
```

---

# 25. Adapter pattern

A capability interface should expose application semantics.

Example:

```python
class PaymentGateway(Protocol):
    async def charge(
        self,
        customer_id: CustomerId,
        amount: Money,
    ) -> Receipt:
        ...
```

Concrete implementation:

```python
class StripePaymentGateway:
    async def charge(
        self,
        customer_id: CustomerId,
        amount: Money,
    ) -> Receipt:
        try:
            ...
        except StripeCardDeclined as exc:
            raise PaymentDeclined(reason=str(exc)) from exc
        except StripeTimeout as exc:
            raise PaymentProviderUnavailable(
                operation="charge",
            ) from exc
```

The rest of the application does not depend on Stripe exception classes.

---

# 26. Bootstrap pattern

Centralize construction.

Conceptual structure:

```python
async def main() -> None:
    settings = load_and_validate_settings()

    async with AsyncExitStack() as stack:
        db = await stack.enter_async_context(
            open_database(settings.database)
        )

        http = await stack.enter_async_context(
            open_http_client(settings.http)
        )

        user_repo = PostgresUserRepo(db)
        customer_gateway = StripeCustomerGateway(http, settings.stripe)

        result = await run_application(
            user_repo=user_repo,
            customer_gateway=customer_gateway,
            ...
        )
```

In `stateless` mode, this bootstrap is where concrete implementations are supplied to required abilities.

Do not instantiate SDK clients deep inside business functions.

---

# 27. Configuration as a capability/value

Avoid repeated:

```python
os.environ["FOO"]
```

inside business code.

Load configuration once:

```python
class Settings(BaseModel):
    database_url: str
    stripe_api_key: SecretStr
```

Then construct:

```text
immutable configuration values
or narrow policy objects
```

Pass only what a component needs.

Do not turn one giant settings object into a service locator.

---

# 28. Clock and randomness

Code that calls:

```python
datetime.now()
time.time()
random.random()
uuid.uuid4()
```

inside domain decisions creates hidden effects.

Extract capabilities when determinism matters.

Examples:

```python
class Clock(Protocol):
    def now(self) -> datetime:
        ...

class IdGenerator(Protocol):
    def new_user_id(self) -> UserId:
        ...
```

Do not abstract time/randomness where it adds no testing or semantic value.

Use judgment.

---

# 29. Mutable state

Find long-lived mutable service fields.

Example:

```python
class Service:
    def __init__(self):
        self.current_user = None
        self.cache = {}
```

Determine whether state is:

```text
request-local
application-global
cache
coordination primitive
domain aggregate state
```

Then give it an owner.

Do not let arbitrary service objects become shared mutable bags.

For concurrency-sensitive state, use proper synchronization.

For domain state, prefer immutable transitions where practical.

---

# 30. Database rules

Database adapters own driver details.

Use transaction scopes explicitly:

```python
async with session.begin():
    ...
```

or the equivalent client API.

During migration, inspect:

```text
transaction boundaries
retry boundaries
unique-constraint mapping
deadlock/serialization failure handling
connection/session lifetime
lazy loading
```

Do not accidentally change transaction semantics while extracting effects.

---

# 31. HTTP/API rules

Routes/controllers are adapters.

They should:

```text
parse
validate
authenticate/authorize at the proper boundary
call application use case
translate typed outcome
return transport response
```

They should not contain the majority of business orchestration.

Example mapping:

```text
EmailAlreadyExists        -> 409
UserNotFound              -> 404
ValidationError           -> 422 / boundary-specific response
PaymentDeclined           -> domain-appropriate response
DatabaseUnavailable       -> 503
unexpected defect         -> 500 + telemetry
```

Do not expose internal exception strings directly to clients.

---

# 32. Worker/job rules

Jobs need the same discipline as HTTP.

A worker boundary should:

```text
decode message
validate
run effect/use case
classify typed failure
decide retry/dead-letter
record telemetry
ack/nack
```

Do not let broker retry behavior substitute for application error classification.

---

# 33. Logging

Do not use logging as error handling.

Bad:

```python
try:
    ...
except Exception:
    logger.exception("failed")
    return None
```

Logging should report an already-understood event/failure.

Avoid logging the same error at every layer.

Prefer context-rich logs at the layer that owns the operation.

---

# 34. Context variables

`contextvars.ContextVar` is useful for dynamic execution context such as:

```text
trace id
request id
tenant context when architecturally justified
```

Do not use it as a general dependency container.

A `ContextVar[Database]` is usually a hidden dependency wearing a nicer jacket.

---

# 35. Testing model

A migrated codebase should become easier to test.

## Pure logic

No mocks needed.

## Effect/use-case tests

Supply fake capabilities.

Example:

```python
class FakeUserRepo:
    def __init__(self, users: dict[str, User]) -> None:
        self.users = users

    async def find_by_email(self, email: str) -> User | None:
        return self.users.get(email)
```

## Adapter tests

Verify driver error translation.

Example:

```text
unique constraint -> EmailAlreadyExists
timeout -> DatabaseUnavailable
```

## Concurrency tests

Verify:

```text
failure propagation
cancellation
timeout
cleanup
no orphan tasks
```

## Property tests

Use Hypothesis where domain invariants benefit from generative testing.

---

# 36. Avoid mock-heavy architecture

If every test requires:

```python
monkeypatch.setattr(...)
patch("app.foo.client")
patch("app.bar.settings")
```

the design probably still has hidden dependencies.

Prefer fake capability implementations supplied explicitly.

Mocks remain acceptable for narrow interaction tests, but should not be the only way to test business logic.

---

# 37. Progressive adoption strategy

Do not require the repository to become 100% effect-based before shipping.

Create an Effect-style core and adapt legacy edges.

Recommended direction:

```text
legacy Python
    ↓
adapter
    ↓
typed application/effect core
    ↓
adapter
    ↓
legacy/framework code
```

Then shrink the legacy zone over time.

Avoid the reverse direction where new effect code depends directly on old global state.

---

# 38. Effect boundary pattern

During migration, legacy code may still raise conventional exceptions.

Create narrow boundaries.

Bad:

```python
# scattered through 50 files
try:
    ...
except SomeSdkException:
    ...
```

Better:

```text
legacy/SDK adapter
    ↓ normalize failures
effect boundary
    ↓ typed failures
application core
```

Centralize normalization per dependency.

---

# 39. Do not wrap every function

An AI migration agent must avoid mechanical transformations like:

```text
every function -> Effect
every return -> success(...)
every raise -> throw(...)
```

That produces ceremony without architecture.

Convert functions when an effect contract adds useful information.

Keep ordinary Python for:

```text
pure transformations
small value-object helpers
deterministic formatting
local collection operations
simple predicates
```

---

# 40. Do not create a custom mini-Effect runtime

Forbidden unless explicitly required:

```text
custom Effect class
custom Layer framework
custom fiber scheduler
custom Result + custom Either
custom cancellation runtime
custom dependency container
custom schema framework
```

Use existing Python primitives/libraries.

The purpose of the migration is to remove accidental complexity.

---

# 41. Do not mix three error paradigms randomly

Bad target architecture:

```text
some functions raise domain exceptions
some return returns.Result
some return custom Result
some use stateless typed errors
some return None
```

Choose a dominant rule.

For this skill, default:

```text
inside Effect-style application layer:
    stateless typed recoverable error channel

inside infrastructure adapter implementation:
    ordinary library exceptions are caught narrowly and normalized

for defects:
    raise normally

for simple absence:
    T | None

at external transport boundary:
    map typed application outcomes to protocol responses
```

---

# 42. Suggested project layering

Use repository conventions if already strong.

Otherwise a practical shape is:

```text
app/
├─ domain/
│  ├─ models.py
│  ├─ errors.py
│  └─ rules.py
│
├─ application/
│  ├─ capabilities.py
│  ├─ effects.py
│  └─ use_cases/
│
├─ infrastructure/
│  ├─ postgres/
│  ├─ stripe/
│  ├─ http/
│  └─ messaging/
│
├─ interfaces/
│  ├─ http/
│  ├─ cli/
│  └─ workers/
│
└─ bootstrap.py
```

Do not reorganize the entire repository merely to match this tree if existing boundaries are already good.

Architecture matters more than folder names.

---

# 43. Refactoring recipe: global DB

## Before

```python
# database.py
db = Database(...)

# users.py
from app.database import db

async def load_user(user_id: str):
    return await db.get(user_id)
```

## After capability extraction

```python
class UserRepo(Protocol):
    async def get(self, user_id: UserId) -> User:
        ...
```

Then the effect/use case requires `UserRepo`.

Concrete DB construction moves to bootstrap.

---

# 44. Refactoring recipe: `ValueError`

## Before

```python
if balance < amount:
    raise ValueError("not enough balance")
```

## After

```python
@dataclass(frozen=True)
class InsufficientFunds(Exception):
    available: Money
    requested: Money
```

If this is a recoverable business rejection, put it in the typed error channel.

If the value is impossible because an invariant was broken earlier, treat it as a defect instead.

---

# 45. Refactoring recipe: ambiguous `None`

## Before

```python
receipt = await charge(card)

if receipt is None:
    return None
```

## After

Define failures:

```text
CardDeclined
GatewayUnavailable
CustomerNotFound
```

Return `Receipt` on success.

Use `None` only if absence is the real domain value.

---

# 46. Refactoring recipe: broad catch

## Before

```python
try:
    user = await repo.load(user_id)
except Exception:
    return None
```

## After

At the adapter:

```text
expected driver timeout
    -> DatabaseUnavailable

expected missing row
    -> UserNotFound or None according to contract

unexpected error
    -> defect propagates
```

Application code catches typed failures only when it can make a decision.

---

# 47. Refactoring recipe: detached task

## Before

```python
asyncio.create_task(send_email(user))
```

## After

Choose explicitly:

### Part of use-case success

Await it.

### Concurrent child work

Run it inside a task group.

### Durable background work

Enqueue a job/message with a durable worker system.

Do not use detached in-process tasks as a fake queue.

---

# 48. Refactoring recipe: inline retry

## Before

```python
for _ in range(3):
    try:
        return await client.fetch()
    except TimeoutError:
        await asyncio.sleep(1)
```

## After

Create a retry policy.

Define:

```text
TimeoutError normalized to UpstreamUnavailable
retry only UpstreamUnavailable
3 attempts
exponential backoff
bounded delay
```

Apply through `stateless` schedule/retry or the selected focused retry mechanism.

---

# 49. Refactoring recipe: environment access

## Before

```python
def send():
    key = os.environ["API_KEY"]
    ...
```

## After

Validate configuration at startup.

Construct the adapter once:

```python
client = ApiClient(api_key=settings.api_key)
```

Application code requires a narrow capability, not `Settings`.

---

# 50. Refactoring recipe: time

## Before

```python
if token.expires_at < datetime.now(timezone.utc):
    ...
```

If this rule needs deterministic testing, extract `Clock`.

Effect-like operation requires `Clock`.

A fake clock makes time-based behavior deterministic.

---

# 51. Refactoring recipe: SDK model leakage

## Before

```python
async def charge(...) -> stripe.PaymentIntent:
    ...
```

## After

Map to your own application/domain type:

```python
@dataclass(frozen=True)
class Receipt:
    provider_id: str
    amount: Money
```

Do not make the entire codebase depend on third-party SDK models.

---

# 52. Refactoring recipe: huge service class

## Before

```python
class UserService:
    # db
    # mail
    # stripe
    # cache
    # permissions
    # analytics
    # 40 methods
```

Split by capability/use-case semantics.

Prefer small interfaces and application functions.

Avoid simply turning `UserService` into one giant `Need[UserService]`.

That hides the same coupling behind an effect abstraction.

---

# 53. Error handling policy

The AI agent should maintain a table such as:

| Error | Origin | Recoverable? | Retryable? | Public? |
|---|---|---:|---:|---:|
| EmailAlreadyExists | domain/db mapping | yes | no | yes |
| DatabaseUnavailable | DB adapter | yes | often | no |
| PaymentDeclined | payment adapter | yes | no | maybe |
| PaymentProviderUnavailable | payment adapter | yes | often | no |
| AssertionError | defect | no | no | no |

Keep this table updated during migration.

---

# 54. Capability inventory

Maintain a repository-level capability inventory.

Example:

| Capability | Used by | Implementation |
|---|---|---|
| `UserRepo` | registration, auth | Postgres |
| `PaymentGateway` | checkout | Stripe |
| `Clock` | tokens, subscriptions | SystemClock |
| `IdGenerator` | registration | UUID |
| `Mailer` | notifications | SES |
| `Audit` | privileged actions | structured log/event sink |

If one capability grows too broad, split it.

---

# 55. Concurrency inventory

For every concurrent operation document:

```text
parent
children
max concurrency
failure mode
sibling cancellation
timeout
resource scope
shutdown behavior
```

Do not assume `asyncio.gather` and an Effect-style concurrent composition have identical semantics.

---

# 56. Migration acceptance criteria

A migrated use case should satisfy all relevant items:

- [ ] Success type is explicit.
- [ ] Recoverable failures are identified.
- [ ] Dependencies are explicit.
- [ ] No new hidden global clients.
- [ ] No `except Exception` inside application logic.
- [ ] Defects are not converted to domain errors.
- [ ] `None` is used only for real absence.
- [ ] External exceptions are normalized at adapters.
- [ ] External input is validated at boundaries.
- [ ] Resources close deterministically.
- [ ] Child tasks have owners.
- [ ] Cancellation is preserved.
- [ ] Retry policy is explicit.
- [ ] Timeout policy is explicit.
- [ ] Concurrency limits are explicit.
- [ ] Tests can supply fake capabilities.
- [ ] Strict type checking passes.
- [ ] No meaningless Effect wrapper was added around pure code.

---

# 57. AI-agent workflow for each module

For each target module, output this analysis before/with the refactor:

```text
Module:
Current responsibilities:
Hidden dependencies:
Current exceptions:
Ambiguous None cases:
External I/O:
Resources opened:
Concurrency:
Retry/timeout:
Untyped inputs/outputs:
Proposed capabilities:
Proposed typed errors:
Functions that should remain pure:
Functions that should become effects:
Bootstrap changes:
Tests to update:
```

Then implement incrementally.

---

# 58. AI-agent workflow for each function

Before converting a function to an Effect-like operation, answer:

```text
1. Is it pure?
2. What external capabilities does it require?
3. Which failures can a caller reasonably recover from?
4. Which exceptions are defects?
5. Does it own resources?
6. Does it spawn concurrency?
7. Does it need timeout/retry?
8. Is any input unvalidated?
```

If the function is pure and has no meaningful recoverable error channel, leave it as ordinary Python.

---

# 59. Migration order inside a package

Prefer:

```text
models
    ↓
errors
    ↓
capability Protocols
    ↓
infrastructure adapters
    ↓
application effects/use-cases
    ↓
bootstrap
    ↓
HTTP/worker/CLI adapters
```

This order minimizes temporary coupling.

---

# 60. Legacy bridge rules

During incremental migration:

- migrated core may call legacy adapters only through narrow wrappers;
- legacy routes may call migrated use cases;
- new code must not introduce new ambient globals;
- new recoverable failures should use the new taxonomy;
- new detached async tasks are forbidden;
- new `dict[str, Any]` in the domain/application layer is forbidden.

Do not wait for the entire repository to migrate before enforcing better rules on new code.

---

# 61. What "Effect-TS-like" means in this skill

It means the Python code should inherit these ideas:

```text
typed success
typed recoverable failure
explicit environment/capabilities
composition
interpretable effects
resource safety
structured concurrency
cancellation awareness
retry/schedule policies
testability through supplied capabilities
```

It does **not** mean Python must reproduce every Effect-TS API.

---

# 62. What not to imitate from Effect-TS

Do not create Python equivalents merely for naming symmetry:

```text
Layer
Fiber
Cause
Exit
Chunk
Stream
Ref
Deferred
Queue
Schema
Context
```

unless the application actually needs those semantics and the chosen Python library provides a clear implementation.

Prefer native Python/AnyIO/contextlib equivalents where they are clearer.

---

# 63. Recommended decision rules

Use:

```text
ordinary pure function
```

for deterministic logic.

Use:

```text
T | None
```

for real optional absence.

Use:

```text
typed effect failure
```

for recoverable application failures.

Use:

```text
raise
```

for defects/programmer errors.

Use:

```text
Need[Protocol]
```

for replaceable operational dependencies in stateless mode.

Use:

```text
context manager
```

for resource lifetime.

Use:

```text
AnyIO task group
```

for child concurrency.

Use:

```text
durable queue/job system
```

for intentionally detached work.

Use:

```text
Pydantic
```

for untrusted external data boundaries.

---

# 64. Package policy

Default additions when justified:

```bash
uv add stateless anyio pydantic
```

Optional:

```bash
uv add dishka
```

Do not automatically modify dependencies if equivalent libraries are already standardized in the repository.

Before adding a dependency:

1. inspect current stack;
2. avoid duplicates;
3. verify Python version compatibility;
4. use repository package manager;
5. add type-check/lint configuration if required.

---

# 65. `Dishka` rule

Dishka is optional.

Use it when the project has genuinely complex:

```text
application scopes
request scopes
worker scopes
nested resource ownership
dependency graph assembly
```

Do not add Dishka merely because Effect-TS has `Layer`.

For many stateless-based applications:

```text
Need[T]
+ explicit bootstrap
+ AsyncExitStack
```

is enough.

---

# 66. `returns` rule

Do not introduce `returns.Result` into the same application layer simply because some developers like it.

If `stateless` is the selected effect model:

```text
recoverable application failures -> stateless error channel
```

Use `Result` only at a narrow boundary where a value-level result has a specific advantage.

Avoid abstraction soup.

---

# 67. Observability rule

Effects should not make observability implicit.

Preserve:

```text
trace context
operation name
structured fields
duration
retry count
failure category
```

Use the repository's tracing/logging standard.

Do not put arbitrary logging in every helper.

---

# 68. Performance rule

Do not assume effect abstraction overhead matters, and do not assume it does not.

For performance-sensitive code:

1. profile current code;
2. refactor architecture;
3. benchmark representative workloads;
4. optimize measured bottlenecks.

Keep tight numeric/data loops as plain Python/native/vectorized code unless effects actually belong there.

---

# 69. Safety rule for automated migration

An AI agent must not perform semantic changes silently.

Escalate/document when refactoring reveals ambiguity such as:

```text
Is this exception recoverable or a defect?
Does None mean not-found or failure?
Should this task outlive the request?
Is this retry safe?
Is this DB call inside one transaction?
Is this global client intentionally process-wide?
```

When repository context answers the question, infer from evidence.

When it cannot be determined, preserve current behavior conservatively and mark the ambiguity.

---

# 70. Required output after each migration batch

Report:

```text
Files changed:
Capabilities introduced:
Typed errors introduced:
Globals removed:
Broad catches removed:
Ambiguous None cases resolved:
Resources made scoped:
Detached tasks removed:
Retry/timeout policies introduced:
Validation boundaries added:
Remaining legacy boundaries:
Known semantic ambiguities:
Type-check status:
Test status:
```

---

# 71. Definition of done

The migration is successful when the Python codebase no longer relies on accidental conventions to
understand effects.

A developer should be able to inspect a use case and answer:

```text
What can this operation do?
What does it need?
How can it fail recoverably?
What does success look like?
Who owns its resources?
Who owns its child tasks?
How is it tested?
```

without searching for imported singletons and reading every `try/except`.

---

# 72. Final directive to the migration AI

Do not convert "basic Python" into "functional-looking Python".

Convert:

```text
implicit dependencies
implicit failures
implicit lifetimes
implicit concurrency
implicit validation
```

into:

```text
explicit capabilities
typed recoverable errors
owned resources
structured concurrency
validated boundaries
```

Use `stateless` when an effect abstraction genuinely improves those contracts.

Keep pure Python pure.

Keep defects loud.

Keep boundaries narrow.

Never replace one form of bullshit with abstraction bullshit.
