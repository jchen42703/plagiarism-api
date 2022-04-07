from fastapi import APIRouter
from backend.routes import basic, predict, data

router = APIRouter()
router.include_router(basic.router, tags=["test"], prefix="/test")
router.include_router(predict.router, tags=["predict"])
router.include_router(data.router, tags=["data"])
