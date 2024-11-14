# This file was auto-generated by Fern from our API Definition.

import datetime as dt
import typing

from ..core.datetime_utils import serialize_datetime
from ..core.pydantic_utilities import deep_union_pydantic_dicts, pydantic_v1
from .model_provider_connection_budget_reset_period import ModelProviderConnectionBudgetResetPeriod
from .model_provider_connection_created_by import ModelProviderConnectionCreatedBy
from .model_provider_connection_organization import ModelProviderConnectionOrganization
from .model_provider_connection_provider import ModelProviderConnectionProvider
from .model_provider_connection_scope import ModelProviderConnectionScope


class ModelProviderConnection(pydantic_v1.BaseModel):
    provider: ModelProviderConnectionProvider
    api_key: typing.Optional[str] = None
    deployment_name: typing.Optional[str] = None
    endpoint: typing.Optional[str] = None
    scope: typing.Optional[ModelProviderConnectionScope] = None
    organization: typing.Optional[ModelProviderConnectionOrganization] = None
    created_by: typing.Optional[ModelProviderConnectionCreatedBy] = None
    created_at: typing.Optional[dt.datetime] = None
    updated_at: typing.Optional[dt.datetime] = None
    is_internal: typing.Optional[bool] = pydantic_v1.Field(default=None)
    """
    Whether the model provider connection is internal, not visible to the user.
    """

    budget_limit: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Budget limit for the model provider connection (null if unlimited)
    """

    budget_last_reset_date: typing.Optional[dt.datetime] = pydantic_v1.Field(default=None)
    """
    Date and time the budget was last reset
    """

    budget_reset_period: typing.Optional[ModelProviderConnectionBudgetResetPeriod] = pydantic_v1.Field(default=None)
    """
    Budget reset period for the model provider connection (null if not reset)
    """

    budget_total_spent: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Tracked total budget spent for the given provider connection within the current budget period
    """

    budget_alert_threshold: typing.Optional[float] = pydantic_v1.Field(default=None)
    """
    Budget alert threshold for the given provider connection
    """

    def json(self, **kwargs: typing.Any) -> str:
        kwargs_with_defaults: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        return super().json(**kwargs_with_defaults)

    def dict(self, **kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
        kwargs_with_defaults_exclude_unset: typing.Any = {"by_alias": True, "exclude_unset": True, **kwargs}
        kwargs_with_defaults_exclude_none: typing.Any = {"by_alias": True, "exclude_none": True, **kwargs}

        return deep_union_pydantic_dicts(
            super().dict(**kwargs_with_defaults_exclude_unset), super().dict(**kwargs_with_defaults_exclude_none)
        )

    class Config:
        frozen = True
        smart_union = True
        extra = pydantic_v1.Extra.allow
        json_encoders = {dt.datetime: serialize_datetime}