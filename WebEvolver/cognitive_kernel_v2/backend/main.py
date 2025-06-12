import uvicorn
import os
import argparse
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from cognitive_kernel.cognitive_kernel import (
    CognitiveKernel,
    FILE_LOCATIONS,
    GLOBAL_DB_LOCATIONS,
    DB_LOCATIONS,
    CHARACTER_POOL_PATH,
    CUSTOMIZED_CHARACTER_POOL_PATH,
    TOP_K_SENTENCES,
)
from typing import List, Optional
import auth as auth
import json
import asyncio
from datetime import datetime
from pydantic import BaseModel
from database import (
    update_or_create_session,
    get_sessions_by_username,
    get_session_by_id,
    archive_session_by_id,
    get_rawdata_by_message_id,
    update_or_create_annotation,
    get_all_annotations,
    get_annotations_by_username_and_date_range,
    get_annotation_counts_by_username,
)
from cognitive_kernel.file_loader import read_file_content
from starlette.websockets import WebSocketDisconnect


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Knowledge Engine Config
    parser.add_argument("--num_local_module", type=int, default=1000)
    parser.add_argument("--local_module_path", type=str, default="/app/Database_local/")
    parser.add_argument("--log_path", type=str, default="/app/Logs/")
    parser.add_argument(
        "--global_config_file_path",
        type=str,
        default="/app/cognitive_kernel/memory_kernel/knowledge_engine_config/debug_global_configs.jsonl",
    )
    parser.add_argument(
        "--default_config_file_path",
        type=str,
        default="/app/cognitive_kernel/memory_kernel/knowledge_engine_config/default_configs.json",
    )

    return parser.parse_args()


args = get_args()

# with open(
#     os.environ.get("KR_SERVICE_IP_FILE", "/app/service_url_config.json"), "r"
# ) as f:
#     SERVICE_URLS = json.load(f)

model_name = os.environ.get("MODEL_NAME", "ck")
ACTIVATE_KE = os.environ.get("ACTIVATE_KE", "True").lower() in ("true", "1", "t")
ACTIVATE_SHORT_FEEDBACK = os.environ.get("ACTIVATE_SHORT_FEEDBACK", "True").lower() in (
    "true",
    "1",
    "t",
)
UPLOAD_DIR = "/app/static/uploads"
MAX_CUSTOMIIZED_CHARACTER = int(os.environ.get("MAX_CUSTOMIIZED_CHARACTER", 1))

service_ip = os.environ.get("SERVICE_IP", "11.255.124.151:8081")

current_cognitive_kernel = CognitiveKernel(
    args,
    model_name=model_name,
    # model_name="gpt-4o-mini",
    service_ip=service_ip,
    gpt_model=os.environ.get("GPT_MODEL", "gpt-4o-mini"),
    critic=os.environ.get("CRITIC_MODEL", "gpt"),
    world_model_ip=os.environ.get("WORLD_MODEL_IP", "11.255.124.151:8081"),
)


class AnnotationModel(BaseModel):
    session_id: str
    message_id: str
    username: str
    old_message: str
    suggestion: str
    annotations: str
    created_time: str
    updated_time: str


app = FastAPI()
app.include_router(auth.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/avatar", StaticFiles(directory="/app/static/avatar"), name="avatars")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.post("/generate_ck")
async def generate_ck(input_info: dict):
    return StreamingResponse(
        current_cognitive_kernel.generate_for_demo(
            messages=input_info["messages"],
            CKStatus=input_info["CKStatus"],
            username=input_info["username"],
            message_id=input_info["currentMessageId"],
        ),
        media_type="application/json",
    )


async def generate_messages(websocket: WebSocket, data_info, mode):
    try:
        async for message in current_cognitive_kernel.generate_for_demo(
            messages=data_info["messages"],
            CKStatus=data_info["CKStatus"],
            username=data_info["username"],
            message_id=data_info["currentMessageId"],
            mode=mode,
        ):
            await asyncio.sleep(0.001)
            await websocket.send_text(message)
        await websocket.send_text("[save_message]")
    except asyncio.CancelledError:
        print("Message generation cancelled.")
        await websocket.send_text("[task_cancelled]")


async def eval_task(websocket: WebSocket):
    try:
        async for message in current_cognitive_kernel.evaluation():
            await asyncio.sleep(0.001)
            await websocket.send_text(message)
        await websocket.send_text("[evaluation_done]")
    except asyncio.CancelledError:
        print("Evaluation cancelled.")
        await websocket.send_text("[evaluation_cancelled]")


@app.websocket("/setup_eval_ws")
async def websocket_eval_endpoint(websocket: WebSocket):
    print("eval websocket connected")
    await websocket.accept()
    evaluation_task = None
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "start":
                    if evaluation_task:
                        evaluation_task.cancel()
                    evaluation_task = asyncio.create_task(eval_task(websocket))
                elif data == "stop" and evaluation_task:
                    evaluation_task.cancel()
                    evaluation_task = None
                    await websocket.send_text("[evaluation_cancelled]")
                else:
                    raise NotImplementedError

            except WebSocketDisconnect:
                print("WebSocket connection closed.")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    except WebSocketDisconnect as e:
        print(f"WebSocket connection closed with code: {e.code}, reason: {e.reason}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close()


@app.websocket("/setup_ws")
async def websocket_endpoint(websocket: WebSocket):
    print("websocket connected")
    await websocket.accept()
    message_task = None
    try:
        while True:
            try:
                data = await websocket.receive_text()
                data_info = json.loads(data)

                print("received data:", data_info)
                if data_info["action"] == "generation":
                    # this is a test for generate_for_demo_search, which performs multiple concurrent actions
                    # 1) callweb / direct answer 2) within callweb, run multiples times concurrently (no search algorithm is applied yet).
                    async for message in current_cognitive_kernel.generate_for_demo(
                        messages=data_info["messages"],
                        CKStatus=data_info["CKStatus"],
                        username=data_info["username"],
                        message_id=data_info["currentMessageId"],
                        mode="generation",
                        # mode="mcts",
                        # mode="concurrent",
                        # mode="concurrent_feedback",
                        # mode="tree_of_thought",
                    ):
                        await asyncio.sleep(0.001)
                        await websocket.send_text(message)
                    await websocket.send_text("[save_message]")
                elif data_info["action"] == "regeneration":
                    if message_task:
                        message_task.cancel()
                    message_task = asyncio.create_task(
                        generate_messages(websocket, data_info, mode="regeneration")
                    )
                elif data_info["action"] == "stop" and message_task:
                    message_task.cancel()
                    message_task = None
                    await websocket.send_text("[task_cancelled]")

            except WebSocketDisconnect:
                print("WebSocket connection closed.")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    finally:
        if message_task:
            message_task.cancel()
        print("WebSocket connection fully closed.")


async def test_ck_inference():
    for i in range(10):
        await asyncio.sleep(0.1)
        yield f"data: {i}\n\n"

# @app.post("/inference_api_close_all_browsers")
# async def close_all_browsers(input_info: dict):
#     import requests
#     url = "http://web:3000/getBrowser"
#     response = requests.post(url)
#     return JSONResponse(content=response, media_type="application/json")

@app.post("/inference_api")
async def inference_api(input_info: dict):
    messages = input_info["messages"]
    print("messages:", messages)
    if "full_info" in input_info:
        response = await current_cognitive_kernel.inference_api(
            messages, input_info["full_info"]
        )
    else:
        response = await current_cognitive_kernel.inference_api(messages)
    return JSONResponse(content=response, media_type="application/json")

@app.post("/inference_api_gaia")
async def inference_api(input_info: dict):
    messages = input_info["messages"]
    print("messages:", messages)
    if "full_info" in input_info:
        full_info = input_info["full_info"]
    else:
        full_info = False
    response = await current_cognitive_kernel.inference_api_gaia_new_workflow(
        messages, full_info, input_info["world_model_type"], (input_info["world_model_search_depth"])
    )

    return JSONResponse(content=response, media_type="application/json")


@app.post("/inference_api_wm")
async def inference_api_wm(input_info: dict):
    messages = input_info["messages"]
    print("input_info", input_info)
    if "world_model_search" in input_info:
        world_model_search = input_info["world_model_search"]
    else:
        world_model_search = None
    print("messages:", messages)
    if "full_info" in input_info:
        response = await current_cognitive_kernel.inference_api_wm(
            messages, world_model_search, int(input_info["search_depth"]), input_info["full_info"]
        )
    else:
        response = await current_cognitive_kernel.inference_api_wm(messages, world_model_search, int(input_info["search_depth"]))
    return JSONResponse(content=response, media_type="application/json")

@app.post("/inference_api_call_web")
async def inference_api_call_web(input_info: dict):
    query = input_info["query"]
    target_url = input_info["target_url"]
    session_id = input_info["session_id"]
    message_id = input_info["message_id"]
    username = input_info["username"]
    max_steps = input_info["max_steps"]
    storage_state = (
        input_info["storage_state"] if "storage_state" in input_info else None
    )
    geo_location = input_info["geo_location"] if "geo_location" in input_info else None
    response = await current_cognitive_kernel.inference_api_call_web(
        query=query,
        target_url=target_url,
        session_id=session_id,
        message_id=message_id,
        username=username,
        max_steps=max_steps,
        storage_state=storage_state,
        geo_location=geo_location,
    )
    return JSONResponse(content=response, media_type="application/json")


@app.post("/inference_api_upload_file")
async def inference_api_upload_file(file: UploadFile = File(...)):

    file_location = f"{FILE_LOCATIONS}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        content = await file.read()
        file_object.write(content)

    if ACTIVATE_KE:
        current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        db_location = f"{DB_LOCATIONS}/{current_timestamp}.{file.filename}.db"
        db_name = f"{current_timestamp}.{file.filename}"

        file_content = read_file_content(
            file_location=file_location, file_name=file.filename
        )

        selected_sentences = list()
        selected_meta_data = list()
        for tmp_instance in file_content[:TOP_K_SENTENCES]:
            selected_sentences.append(tmp_instance["text"])
            selected_meta_data.append(tmp_instance["meta"])

        current_cognitive_kernel.update_knowledge_engine(
            db_name=db_name,
            db_path=db_location,
            sentences=selected_sentences,
            metadata=selected_meta_data,
        )
    else:
        db_location = ""
        db_name = ""

    uploaded_info = {
        "file_location": file_location,
        "db_location": db_location,
        "file_name": file.filename,
        "db_name": db_name,
    }
    return JSONResponse(content=uploaded_info, media_type="application/json")


@app.post("/inference_api_history")
async def inference_api_history(input_info: dict):
    print(input_info)
    candidate_history_messages = input_info["candidate_history_messages"]
    target_query = input_info["target_query"]
    retrieved_info = await current_cognitive_kernel.inference_api_history_retrieval(
        candidate_history_messages, target_query
    )
    return JSONResponse(content=retrieved_info, media_type="application/json")


@app.get("/start_evaluation")
async def start_evaluation(username: str):
    return StreamingResponse(
        current_cognitive_kernel.evaluation(),
        media_type="application/json",
    )


@app.post("/clean_up_ck")
async def clean_up_ck(input_info: dict):
    current_cognitive_kernel.clean_up(
        CKStatus=input_info["CKStatus"],
        username=input_info["username"],
    )
    return JSONResponse(content={"data": "success"}, media_type="application/json")


@app.post("/retrieve_history")
async def retrieve_history(input_info: dict):
    print(f"input_info: {input_info}")
    sessions_info = get_sessions_by_username(
        model_name=input_info["model_name"], username=input_info["username"]
    )
    return JSONResponse(content={"data": sessions_info}, media_type="application/json")


@app.post("/retrieve_message_session_by_id")
async def retrieve_message_session_by_id(input_info: dict):
    session_info = get_session_by_id(session_id=input_info["session_id"])
    return JSONResponse(content={"data": session_info}, media_type="application/json")


@app.post("/archive_message_session_by_id")
async def archive_message_session_by_id(input_info: dict):
    try:
        archive_session_by_id(session_id=input_info["session_id"])
        return JSONResponse(
            content={"status": "success"}, media_type="application/json"
        )
    except Exception as e:
        return JSONResponse(
            content={"status": "failed", "error": str(e)},
            status_code=500,
            media_type="application/json",
        )


@app.post("/save_message_to_db")
async def save_message_to_db(input_info: dict):
    try:
        update_or_create_session(
            session_id=input_info["session_id"],
            username=input_info["username"],
            model_name=input_info["model_name"],
            messages=input_info["messages"],
            updated_time=input_info["updated_time"],
        )
        return JSONResponse(content={"data": "success"}, media_type="application/json")
    except Exception as e:
        return JSONResponse(
            content={"data": "failed", "error": str(e)},
            status_code=500,
            media_type="application/json",
        )


@app.post("/retrieve_rawdata_by_message_id")
async def retrieve_rawdata_by_message_id(input_info: dict):
    rawdata_info = get_rawdata_by_message_id(message_id=input_info["currentMessageId"])
    return JSONResponse(content={"data": rawdata_info}, media_type="application/json")


@app.post("/submit_annotation")
async def submit_annotation(input_info: dict):
    print("received annotation:", input_info)
    try:
        update_or_create_annotation(
            session_id=input_info["session_id"],
            message_id=input_info["currentMessageId"],
            username=input_info["username"],
            tag=input_info["tag"],
            for_evaluation=input_info["for_evaluation"],
            old_message=input_info["oldKnowledge"],
            suggestion=input_info["Suggestion"],
            messages_in_train_format=input_info["messages_in_train_format"],
            updated_time=input_info["updated_time"],
        )
        current_cognitive_kernel.update_online_feedback_db(input_info)
        return JSONResponse(content={"data": "success"}, media_type="application/json")
    except Exception as e:
        return JSONResponse(
            content={"data": "failed", "error": str(e)},
            status_code=500,
            media_type="application/json",
        )


@app.get("/download_annotations", response_model=List[AnnotationModel])
async def download_annotations(
    username: str, start_date: str, end_date: str, download_type: str
):
    if download_type == "all":
        annotations = get_all_annotations()
    else:
        annotations = get_annotations_by_username_and_date_range(
            username, start_date, end_date
        )
    return annotations


@app.get("/annotation_statistics")
async def annotation_statistics(username: str, current_time: str):
    annotation_statistics = get_annotation_counts_by_username(username, current_time)
    return JSONResponse(
        content={"data": annotation_statistics}, media_type="application/json"
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), ckStatus: str = Form(...)):
    ck_status_data = json.loads(ckStatus)

    file_location = f"{FILE_LOCATIONS}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        content = await file.read()
        file_object.write(content)

    if ACTIVATE_KE:
        db_location = (
            f"{DB_LOCATIONS}/{ck_status_data['session_id']}_{file.filename}.db"
        )
        db_name = f"{file.filename}_{ck_status_data['session_id']}"

        file_content = read_file_content(
            file_location=file_location, file_name=file.filename
        )

        selected_sentences = list()
        selected_meta_data = list()
        for tmp_instance in file_content[:TOP_K_SENTENCES]:
            selected_sentences.append(tmp_instance["text"])
            selected_meta_data.append(tmp_instance["meta"])

        current_cognitive_kernel.update_knowledge_engine(
            db_name=db_name,
            db_path=db_location,
            sentences=selected_sentences,
            metadata=selected_meta_data,
        )

    return {"info": f"file '{file.filename}' saved at '{file_location}'"}


@app.post("/upload_character_info")
async def upload_avatar(file: UploadFile = File(...)):
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"{UPLOAD_DIR}/{unique_filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    file_url = f"/api/uploads/{unique_filename}"
    return JSONResponse(content={"url": file_url})


@app.post("/delete_agent/")
async def delete_agent(username: str = Form(...), agent_name: str = Form(...)):
    if os.path.exists(f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_name}"):
        shutil.rmtree(f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_name}")
        current_cognitive_kernel.delete_character(f"{username}_{agent_name}")
        return {"info": "success"}
    else:
        raise HTTPException(status_code=409, detail="Agent does not exist.")


@app.post("/create_agent/")
async def create_agent(
    username: Optional[str] = Form(None),
    avatarURL: Optional[str] = File(None),
    backgroundURL: Optional[str] = File(None),
    agent_name: str = Form(...),
    agent_id: str = Form(...),
    description: Optional[str] = Form(None),
):
    if username is None:
        raise HTTPException(status_code=409, detail="username is required.")
    if os.path.exists(f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}"):
        raise HTTPException(status_code=409, detail="Agent already exists.")
    existing_agent_count = 0
    for tmp_file in os.listdir(CUSTOMIZED_CHARACTER_POOL_PATH):
        if username == tmp_file.split("_")[0]:
            existing_agent_count += 1
    if existing_agent_count >= MAX_CUSTOMIIZED_CHARACTER:
        raise HTTPException(
            status_code=409,
            detail=f"Exceeds the maximum number of customized characters: {MAX_CUSTOMIIZED_CHARACTER}",
        )
    os.mkdir(f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}")
    os.mkdir(f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}/functions")
    if backgroundURL and len(backgroundURL) > 0:
        real_background_path = backgroundURL.replace("/api/uploads", UPLOAD_DIR)
        db_location = f"{GLOBAL_DB_LOCATIONS}/{username}_{agent_id}_background.db"
        db_name = f"{username}_{agent_id}"
        file_content = read_file_content(
            file_location=real_background_path,
            file_name=f"{username}_{agent_id}_background.txt",
        )
        selected_sentences = list()
        selected_meta_data = list()
        for tmp_instance in file_content[:TOP_K_SENTENCES]:
            selected_sentences.append(tmp_instance["text"])
            selected_meta_data.append(tmp_instance["meta"])
        print(f"selected_sentences: {selected_sentences}")
        print(f"selected_meta_data: {selected_meta_data}")
        current_cognitive_kernel.update_knowledge_engine(
            db_name=db_name,
            db_path=db_location,
            sentences=selected_sentences,
            metadata=selected_meta_data,
        )
    avatar_location = (
        f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}/avatar.png"
    )
    if avatarURL and len(avatarURL) > 0:
        real_avatar_path = avatarURL.replace("/api/uploads", UPLOAD_DIR)
        shutil.copy(
            real_avatar_path,
            avatar_location,
        )
    else:
        shutil.copy(
            "/app/resources/default_avatar.png",
            avatar_location,
        )

    agent_info = {}
    agent_info["name"] = f"{username}_{agent_id}"
    agent_info["id"] = agent_id
    agent_info["shown_title"] = f"{agent_name}"
    agent_info["description"] = "Your Customized Character"
    agent_info["visible_users"] = [username]
    agent_info["head_system_prompt"] = description
    agent_info["tail_system_prompt"] = (
        "Please play your role well, do not violate or reveal your character. When the player behaves maliciously or provokes you many times, please politely refuse and express anger"
    )
    agent_info["global_db_info"] = {
        f"{username}_{agent_id}": f"{username}_{agent_id}_background"
    }
    agent_info["system_prompt_sequence"] = list()
    agent_info["system_prompt_sequence"].append(
        {
            "name": "available_functions",
            "pre_defined": True,
            "step_type": "dynamic",
            "head": "",
            "content": [],
            "CK_status_key": "",
        }
    )
    agent_info["system_prompt_sequence"].append(
        {
            "name": "character_dbs",
            "pre_defined": False,
            "step_type": "static",
            "head": "可用的db名称及对应描述如下:",
            "content": [f"{username}_{agent_id}"],
            "CK_status_key": "uploaded_files",
        }
    )

    shutil.copytree(
        f"{CHARACTER_POOL_PATH}/cognitiveKernel/functions/CallMemoryKernel",
        f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}/functions/CallMemoryKernel",
    )

    with open(
        f"{CUSTOMIZED_CHARACTER_POOL_PATH}/{username}_{agent_id}/{os.environ.get('MODEL_NAME', '')}_info.json", "w"
    ) as f:
        json.dump(agent_info, f)
    current_cognitive_kernel.update_character(f"{username}_{agent_id}", "customized")

    return {"info": "success"}


@app.post("/get_all_characters")
async def get_all_characters(username: str):
    return JSONResponse(current_cognitive_kernel.get_all_characters(username=username))


num_worker = int(os.environ.get("NUM_WORKERS", 16))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=num_worker)
