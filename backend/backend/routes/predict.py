from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import pandas as pd

from backend.api.prediction import initialize_pipeline

router = APIRouter()
cat_pipeline = initialize_pipeline("./config.yml", "catboost")
lgb_pipeline = initialize_pipeline("./config.yml", "lightgbm")


class RawData(BaseModel):
    building_id: int
    geo_level_1_id: int
    geo_level_2_id: int
    geo_level_3_id: int
    count_floors_pre_eq: int
    age: int
    area_percentage: int
    height_percentage: int
    # Arbitrary codes
    land_surface_condition: str
    foundation_type: str
    roof_type: str
    ground_floor_type: str
    other_floor_type: str
    position: str
    plan_configuration: str

    # Superstructure (1/0)
    has_superstructure_adobe_mud: int
    has_superstructure_stone_flag: int
    has_superstructure_mud_mortar_stone: int
    has_superstructure_mud_mortar_brick: int
    has_superstructure_cement_mortar_brick: int
    has_superstructure_timber: int
    has_superstructure_bamboo: int
    has_superstructure_rc_non_engineered: int
    has_superstructure_rc_engineered: int
    has_superstructure_other: int

    legal_ownership_status: str
    count_families: int

    # Secondary use
    has_secondary_use: int  # if this is 0, everything else is 0
    has_secondary_use_agriculture: int
    has_secondary_use_hotel: int
    has_secondary_use_rental: int
    has_secondary_use_institution: int
    has_secondary_use_school: int
    has_secondary_use_industry: int
    has_secondary_use_health_post: int
    has_secondary_use_gov_office: int
    has_secondary_use_use_police: int
    has_secondary_use_other: int

    class Config:
        schema_extra = {
            "example": {
                "building_id": 802906,
                "geo_level_1_id": 6,
                "geo_level_2_id": 487,
                "geo_level_3_id": 12198,
                "count_floors_pre_eq": 2,
                "age": 30,
                "area_percentage": 6,
                "height_percentage": 5,
                "land_surface_condition": "t",
                "foundation_type": "r",
                "roof_type": "n",
                "ground_floor_type": "f",
                "other_floor_type": "q",
                "position": "t",
                "plan_configuration": "d",
                "has_superstructure_adobe_mud": 1,
                "has_superstructure_mud_mortar_stone": 1,
                "has_superstructure_stone_flag": 0,
                "has_superstructure_cement_mortar_stone": 0,
                "has_superstructure_mud_mortar_brick": 0,
                "has_superstructure_cement_mortar_brick": 0,
                "has_superstructure_timber": 0,
                "has_superstructure_bamboo": 0,
                "has_superstructure_rc_non_engineered": 0,
                "has_superstructure_rc_engineered": 0,
                "has_superstructure_other": 0,
                "legal_ownership_status": "v",
                "count_families": 1,
                "has_secondary_use": 0,
                "has_secondary_use_agriculture": 0,
                "has_secondary_use_hotel": 0,
                "has_secondary_use_rental": 0,
                "has_secondary_use_institution": 0,
                "has_secondary_use_school": 0,
                "has_secondary_use_industry": 0,
                "has_secondary_use_health_post": 0,
                "has_secondary_use_gov_office": 0,
                "has_secondary_use_use_police": 0,
                "has_secondary_use_other": 0,
            }
        }


class PredictReq(BaseModel):
    """For post /api/predict"""

    model_type: str
    data: RawData

    class Config:
        schema_extra = {
            "example": {
                "model_type": "lightgbm",
                "data": {
                    "building_id": 802906,
                    "geo_level_1_id": 6,
                    "geo_level_2_id": 487,
                    "geo_level_3_id": 12198,
                    "count_floors_pre_eq": 2,
                    "age": 30,
                    "area_percentage": 6,
                    "height_percentage": 5,
                    "land_surface_condition": "t",
                    "foundation_type": "r",
                    "roof_type": "n",
                    "ground_floor_type": "f",
                    "other_floor_type": "q",
                    "position": "t",
                    "plan_configuration": "d",
                    "has_superstructure_adobe_mud": 1,
                    "has_superstructure_mud_mortar_stone": 1,
                    "has_superstructure_stone_flag": 0,
                    "has_superstructure_cement_mortar_stone": 0,
                    "has_superstructure_mud_mortar_brick": 0,
                    "has_superstructure_cement_mortar_brick": 0,
                    "has_superstructure_timber": 0,
                    "has_superstructure_bamboo": 0,
                    "has_superstructure_rc_non_engineered": 0,
                    "has_superstructure_rc_engineered": 0,
                    "has_superstructure_other": 0,
                    "legal_ownership_status": "v",
                    "count_families": 1,
                    "has_secondary_use": 0,
                    "has_secondary_use_agriculture": 0,
                    "has_secondary_use_hotel": 0,
                    "has_secondary_use_rental": 0,
                    "has_secondary_use_institution": 0,
                    "has_secondary_use_school": 0,
                    "has_secondary_use_industry": 0,
                    "has_secondary_use_health_post": 0,
                    "has_secondary_use_gov_office": 0,
                    "has_secondary_use_use_police": 0,
                    "has_secondary_use_other": 0,
                },
            }
        }


@router.post("/predict")
async def predict(req: PredictReq) -> JSONResponse:
    data = req.data
    model_type = req.model_type
    if model_type == "catboost":
        confidences = cat_pipeline.predict(pd.DataFrame(data.dict(), index=[0]))
    elif model_type == "lightgbm":
        confidences = lgb_pipeline.predict(pd.DataFrame(data.dict(), index=[0]))
    else:
        raise HTTPException(
            status_code=400,
            detail={"msg": f"model_type {model_type} is not supported."},
        )

    pred = confidences.argmax(axis=1) + 1
    return {"confidences": confidences.tolist(), "prediction": pred.tolist()}


@router.get("/num_feats")
async def num_feats(model_type: str) -> JSONResponse:
    if model_type == "catboost":
        num_features = cat_pipeline.model.num_features()
    elif model_type == "lightgbm":
        num_features = lgb_pipeline.model.num_features()
    else:
        raise HTTPException(
            status_code=400,
            detail={"msg": f"model_type {model_type} is not supported."},
        )

    return {"num_features": num_features}
