# This file was auto-generated by Fern from our API Definition.

import typing

from label_studio_sdk.client import AsyncLabelStudio, LabelStudio
from .utilities import validate_response


async def test_list_(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    expected_response: typing.Any = [
        {
            "id": 1,
            "organization": 1,
            "project": 1,
            "url": "url",
            "send_payload": True,
            "send_for_all_actions": True,
            "headers": {"key": "value"},
            "is_active": True,
            "actions": ["PROJECT_CREATED"],
            "created_at": "2024-01-15T09:30:00Z",
            "updated_at": "2024-01-15T09:30:00Z",
        }
    ]
    expected_types: typing.Any = (
        "list",
        {
            0: {
                "id": "integer",
                "organization": "integer",
                "project": "integer",
                "url": None,
                "send_payload": None,
                "send_for_all_actions": None,
                "headers": ("dict", {0: (None, None)}),
                "is_active": None,
                "actions": ("list", {0: None}),
                "created_at": "datetime",
                "updated_at": "datetime",
            }
        },
    )
    response = client.webhooks.list()
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.webhooks.list()
    validate_response(async_response, expected_response, expected_types)


async def test_create(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    expected_response: typing.Any = {
        "id": 1,
        "organization": 1,
        "project": 1,
        "url": "url",
        "send_payload": True,
        "send_for_all_actions": True,
        "headers": {"key": "value"},
        "is_active": True,
        "actions": ["PROJECT_CREATED"],
        "created_at": "2024-01-15T09:30:00Z",
        "updated_at": "2024-01-15T09:30:00Z",
    }
    expected_types: typing.Any = {
        "id": "integer",
        "organization": "integer",
        "project": "integer",
        "url": None,
        "send_payload": None,
        "send_for_all_actions": None,
        "headers": ("dict", {0: (None, None)}),
        "is_active": None,
        "actions": ("list", {0: None}),
        "created_at": "datetime",
        "updated_at": "datetime",
    }
    response = client.webhooks.create(url="url")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.webhooks.create(url="url")
    validate_response(async_response, expected_response, expected_types)


async def test_info(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.webhooks.info() is None  # type: ignore[func-returns-value]

    assert await async_client.webhooks.info() is None  # type: ignore[func-returns-value]


async def test_get(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    expected_response: typing.Any = {
        "id": 1,
        "organization": 1,
        "project": 1,
        "url": "url",
        "send_payload": True,
        "send_for_all_actions": True,
        "headers": {"key": "value"},
        "is_active": True,
        "actions": ["PROJECT_CREATED"],
        "created_at": "2024-01-15T09:30:00Z",
        "updated_at": "2024-01-15T09:30:00Z",
    }
    expected_types: typing.Any = {
        "id": "integer",
        "organization": "integer",
        "project": "integer",
        "url": None,
        "send_payload": None,
        "send_for_all_actions": None,
        "headers": ("dict", {0: (None, None)}),
        "is_active": None,
        "actions": ("list", {0: None}),
        "created_at": "datetime",
        "updated_at": "datetime",
    }
    response = client.webhooks.get(id=1)
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.webhooks.get(id=1)
    validate_response(async_response, expected_response, expected_types)


async def test_delete(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    # Type ignore to avoid mypy complaining about the function not being meant to return a value
    assert client.webhooks.delete(id=1) is None  # type: ignore[func-returns-value]

    assert await async_client.webhooks.delete(id=1) is None  # type: ignore[func-returns-value]


async def test_update(client: LabelStudio, async_client: AsyncLabelStudio) -> None:
    expected_response: typing.Any = {
        "id": 1,
        "organization": 1,
        "project": 1,
        "url": "url",
        "send_payload": True,
        "send_for_all_actions": True,
        "headers": {"key": "value"},
        "is_active": True,
        "actions": ["PROJECT_CREATED"],
        "created_at": "2024-01-15T09:30:00Z",
        "updated_at": "2024-01-15T09:30:00Z",
    }
    expected_types: typing.Any = {
        "id": "integer",
        "organization": "integer",
        "project": "integer",
        "url": None,
        "send_payload": None,
        "send_for_all_actions": None,
        "headers": ("dict", {0: (None, None)}),
        "is_active": None,
        "actions": ("list", {0: None}),
        "created_at": "datetime",
        "updated_at": "datetime",
    }
    response = client.webhooks.update(id_=1, url="url", webhook_serializer_for_update_url="url")
    validate_response(response, expected_response, expected_types)

    async_response = await async_client.webhooks.update(id_=1, url="url", webhook_serializer_for_update_url="url")
    validate_response(async_response, expected_response, expected_types)
