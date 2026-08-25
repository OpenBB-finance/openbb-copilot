# OpenBB Ada

The service is that backend service that powers the OpenBB Ada (formerly OpenBB
Copilot) agent that is included as the default agent in OpenBB Workspace.

The agent makes use of the same protocol described by the [OpenBB AI SDK](https://github.com/openbb-finance/openbb-ai-sdk).
While we are in the process of migrating over to the helpers, the API models
are shared (this project depends on `openbb-ai`).

## Table of Contents

- [Model Configuration](#model-configuration)
  - [Main Model](#main-model)
  - [Embedding Model](#embedding-model)
  - [Search Model](#search-model)
- [Building for Production](#building-for-production)
- [Development](#development)
  - [Setup Your Environment](#setup-your-environment)
  - [Running the Service](#running-the-service)
  - [Linting and Formatting](#linting-and-formatting)
  - [Running Tests](#running-tests)

## Model Configuration

The service supports flexible model configuration, allowing you to use different providers and models for different purposes. You can mix and match providers - for example, use Claude via OpenRouter for main reasoning while using OpenAI for embeddings.

### Main Model

#### OPENBB_AGENT_MODEL_PROVIDER

This variable determines how the service connects to LLM providers. Set this to match your infrastructure:

| Provider Value | When to Use | Notes |
|----------------|-------------|-------|
| `openai` | Using OpenAI API or any OpenAI-compatible endpoint | Default; works with OpenAI, OpenRouter, Anthropic, etc. |
| `azure` | Using Azure OpenAI Service | Requires Azure-specific configuration |
| `snowflake` | Using Snowflake Cortex AI | For Snowflake-integrated deployments |

#### OPENBB_AGENT_OPENAI_BASE_URL

When `OPENBB_AGENT_MODEL_PROVIDER=openai`, this URL determines which service handles your API requests. You can use any OpenAI-compatible endpoint:

| Service | Base URL | Model Examples |
|---------|----------|----------------|
| **OpenAI** | `https://api.openai.com/v1` | `gpt-4.1`, `gpt-4.1-mini` |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `anthropic/claude-3.5-sonnet`, `google/gemini-pro` |
| **Anthropic** | `https://api.anthropic.com/v1/` | `claude-3-opus`, `claude-3-sonnet` |

#### OPENBB_AGENT_OPENAI_API_KEY

Set the authentication key for your chosen service:

| Variable | Description | Example Values |
|----------|-------------|----------------|
| `OPENBB_AGENT_OPENAI_API_KEY` | Authentication key for your chosen service | Your API key |

#### Model Selection - OPENBB_AGENT_MODEL_

Once you've configured the provider, base URL, and API key, specify which models to use for different tasks:

| Variable | Purpose | Example Values |
|----------|---------|----------------|
| `OPENBB_AGENT_MODEL_MAIN` | Complex reasoning, SQL generation, document Q&A | `gpt-4.1`, `anthropic/claude-3.5-sonnet` |
| `OPENBB_AGENT_MODEL_SMALL` | Simple tasks: titles, summaries, metadata | `gpt-4.1-mini`, `anthropic/claude-3-haiku` |
| `OPENBB_AGENT_MODEL_VISION` | Image analysis and multimodal tasks | `gpt-4.1`, `anthropic/claude-3.5-sonnet` |

### Embedding Model

Configure the model used for text embeddings (RAG/vector search):

| Variable | Description | Example Values |
|----------|-------------|----------------|
| `OPENBB_EMBEDDING_MODEL_PROVIDER` | Embedding provider | `openai` |
| `OPENBB_EMBEDDING_BASE_URL` | Embedding API URL | `https://api.openai.com/v1` |
| `OPENBB_EMBEDDING_API_KEY` | Embedding API key | Your API key |
| `OPENBB_EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |

### Search Model

Configure the model used for web search operations:

| Variable | Description | Example Values |
|----------|-------------|----------------|
| `OPENBB_SEARCH_PROVIDER` | Search provider | `openai` |
| `OPENBB_SEARCH_BASE_URL` | Search API URL | `https://api.openai.com/v1` |
| `OPENBB_SEARCH_API_KEY` | Search API key | Your API key |
| `OPENBB_SEARCH_MODEL` | Search model | `gpt-4.1` |

## Building for Production

To build the production Docker image:

```sh
docker build -t openbb-ada .
```

For Snowflake deployments, if the environment file has changed, you must clear the Docker build cache before rebuilding:

```sh
docker system prune --all
```

This is required because the Docker layer that copies files to `/tmp` doesn't track changes to environment files, which can result in a cached version of the codebase being used instead of the updated one.

## Development

### Setup Your Environment

Make a copy of the example `.env` file and fill in the appropriate values:

```sh
cp .env.example .env
```

The `.env` is be used during both local development and testing.

If there are any values you're unsure about or unable to access, you reach out
to any of the maintainers of this project.

### Running the Service

#### Preferred: Use an IDE Debugger

The preferred local development workflow is to run the service from your IDE
debugger.

Shared debugger configurations are available in:

- `.vscode/launch.json` for VS Code
- `.zed/debug.json` for Zed

Both launch configurations run `uvicorn openbb_ada.main:app` with reload
enabled. The Zed task explicitly loads `.env`; the VS Code launcher uses the
existing project debug configuration as checked in.

#### VS Code

Use the `OpenBB Copilot` launch configuration.

#### Zed

Use the `OpenBB Copilot Z` debug task.

#### Running Locally Without an IDE

If you are not using an IDE debugger, run `uvicorn` directly:

```sh
poetry install --no-root  # install dependencies
uvicorn openbb_ada.main:app --reload --env-file .env --port 7778  # run the service
```

Note: the project relies on `poppler` to convert PDFs to text. The containers have this
dependency preinstalled. If running locally on MacOS and you have `brew` installed, you can
install it with `brew install poppler`.

### Linting and Formatting

We use `ruff` for linting and formatting, which is enforced on the build server.

To format your code, run:

```sh
ruff format
```

To lint and attempt to fix any issues, run:

```sh
ruff check --fix
```

To automate formatting and linting on each commit, install the pre-commit hook:

```sh
pre-commit install
```

### Running Tests

We use `pytest` as our test runner. After running `poetry install`, tests can be
executed locally with:

``` sh
./venv/bin/pytest -n 8 --reruns 2 tests  # execute tests
```

The repository still defines `not stateful` and `stateful` markers, but the `stateful`
tests are skipped integration tests that are being rethought. Do not treat a
separate `-k "stateful"` run as part of the normal local workflow unless that
test path is revived.

Tests require you to have a working `.env` file, which will be loaded by
`pytest-dotenv` when running tests.

We make use of `pytest-xdist` to execute tests in parallel, which speeds up
testing. This means that every test must be written in such a way that it can be
executed in parallel without any issues (for example, not relying on state or
the order of execution). Keep this in mind when writing your own tests. Use the
`-n` flag to specify the number of test workers to use.

We also make use of `pytest-rerunfailures` to rerun failed tests, which is
useful for tests that interface with LLMs (which can be occasionally flaky). Use
the `--reruns` flag to specify the number of times to allow tests to run. We consider
2 tries for our tests as acceptable (in other words, the LLM can get it wrong
_once_). Any test that fails twice in a row means we need to improve our
prompting or evaluation criteria.

It's important to set the environment variable `ENVIRONMENT=TEST` to make sure
services are initiated in testing mode.
