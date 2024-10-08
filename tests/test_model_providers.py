# This file was auto-generated by Fern from our API Definition.

import typing

from label_studio_sdk.client import AsyncLabelStudio, LabelStudio

from .utilities import validate_response


async def test_create(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    expected_response: typing.Any = {
        "provider": "OpenAI",
        "api_key": "api_key",
        "deployment_name": "deployment_name",
        "endpoint": "endpoint",
        "scope": "Organization",
        "organization": 1,
        "created_by": 1,
        "created_at": "2024-01-15T09:30:00Z",
        "updated_at": "2024-01-15T09:30:00Z",
    }
    expected_types: typing.Any = {
        "provider": None,
        "api_key": None,
        "deployment_name": None,
        "endpoint": None,
        "scope": None,
        "organization": "integer",
        "created_by": "integer",
        "created_at": "datetime",
        "updated_at": "datetime",
    }
    response = client.model_providers.create(provider="OpenAI")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.model_providers.create(provider="OpenAI")
    validate_response(async_response, expected_response, expected_types)
