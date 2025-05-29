import asyncio
import logging
import os
import random
import tempfile
import traceback
import uuid
import json
import base64
import queue
import inspect
from io import BytesIO
from collections import defaultdict
from typing import Dict, Any

import aiohttp
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from diffusers import DiffusionPipeline
#from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Add console handler
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logger level to INFO

# Set diffusers logger to INFO level
logging.getLogger("diffusers").setLevel(logging.INFO)

import gc
import torch

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def bytes_to_giga_bytes(bytes):
  return round(bytes / 1024 / 1024 / 1024, 2)

def logVram(des: str):
    logger.info(f'{des} : {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB')

def logComponentMemory(device_map: dict):
    total_memory = 0
    logger.info("Memory usage by component:")
    for component, gpu_id in device_map.items():
        if isinstance(gpu_id, int):
            # Get memory allocated for this GPU
            memory = torch.cuda.max_memory_allocated(gpu_id)
            memory_gb = bytes_to_giga_bytes(memory)
            total_memory += memory
            logger.info(f"  {component} (cuda:{gpu_id}): {memory_gb:.2f} GB")
    logger.info(f"Total memory across all GPUs: {bytes_to_giga_bytes(total_memory)} GB")

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None


class HttpClient:
    session: aiohttp.ClientSession = None

    def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        await self.session.close()
        self.session = None

    def __call__(self) -> aiohttp.ClientSession:
        assert self.session is not None
        return self.session


class TextToImagePipeline:
    pipeline: DiffusionPipeline = None
    device: str = None

    def start(self):
        if torch.cuda.is_available():
            model_path = os.getenv("MODEL_PATH", "/gm-models/CogView4-6B")
            logger.info("Loading CUDA")
            self.device = "cuda"
            # self.pipeline = DiffusionPipeline.from_pretrained(
            #     model_path,
            #     torch_dtype=torch.bfloat16,
            #     device_map="balanced",
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     text_encoder_config={"max_position_embeddings": 1024}
            # )
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="balanced"
            )
            logger.info(self.pipeline.hf_device_map)
            logComponentMemory(self.pipeline.hf_device_map)

        elif torch.backends.mps.is_available():
            model_path = os.getenv("MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

logger.setLevel(logging.INFO)
http_client = HttpClient()
shared_pipeline = TextToImagePipeline()
@asynccontextmanager
async def lifespan(app: FastAPI):
    http_client.start()
    shared_pipeline.start()
    yield
    http_client.stop()
app = FastAPI(lifespan=lifespan)

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"]  # Expose all headers
)

service_url = os.getenv("SERVICE_URL", "http://localhost:8000")
image_dir = os.path.join(".","images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
app.mount("/images", StaticFiles(directory=image_dir), name="images")

# Mount static files for the HTML client
static_dir = os.path.join(".", "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/index.html", StaticFiles(directory=static_dir, html=True), name="static")

def save_image(image, requestid:int, step:int) -> None :
    if logger.level == logging.DEBUG:
        dir = os.path.join(image_dir, str(requestid))
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = "draw" + str(uuid.uuid4()).split("-")[0] + "-" + str(step) + ".png"
        image_path = os.path.join(dir, filename)
        # write image to disk at image_path
        logger.debug(f"Saving image to {image_path}")
        image.save(image_path)
    return None



@app.get("/")
@app.post("/")
@app.options("/")
async def base():
    return "Welcome to Diffusers! Where you can use diffusion models to generate images"

def image_to_base64(image) -> bytes:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_next_request_id() -> int:
    return random.randint(1, 1000000)

def callback_fn(pipe: DiffusionPipeline, step: int, timestep: int, callback_kwargs):
    # Convert latents to image
    request_id = pipe.x
    latents: torch.FloatTensor = callback_kwargs['latents']
    q : queue.Queue = pipe.q
    with torch.no_grad():
        image = shared_pipeline.pipeline.vae.decode(latents / shared_pipeline.pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = shared_pipeline.pipeline.image_processor.postprocess(image)
        # Convert to base64
        save_image(image[0], request_id, step)
        try:
            image_base64 = image_to_base64(image[0])
            data = {'step': step, 'total_step': pipe.num_timesteps, 'image': image_base64}
            q.put(data)
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
    return callback_kwargs


@app.post("/v1/images/generations")
async def generate_image(image_input: TextToImageInput):
    try:
        loop = asyncio.get_event_loop()
        scheduler = shared_pipeline.pipeline.scheduler.from_config(shared_pipeline.pipeline.scheduler.config)
        # Get the actual class of the pipeline
        pipeline_class = type(shared_pipeline.pipeline)
        # Get the from_pipe method from the actual class
        from_pipe_method = getattr(pipeline_class, 'from_pipe')
        # Create new pipeline instance using the correct class's from_pipe method
        pipeline = from_pipe_method(shared_pipeline.pipeline, scheduler=scheduler)
        # Ensure the pipeline is using bfloat16
        pipeline = pipeline.to(dtype=torch.bfloat16)
        generator = torch.Generator(device=shared_pipeline.device)
        generator.manual_seed(random.randint(0, 10000000))
        pipeline.q  = queue.Queue()
        pipeline.x = get_next_request_id()

        async def generate():
            try:
                loop.run_in_executor(
                    None, 
                    lambda: pipeline(
                        image_input.prompt, 
                        generator=generator, 
                        num_inference_steps=image_input.n,
                        callback_on_step_end=callback_fn,
                        callback_on_step_end_tensor_inputs=['latents']
                    )
                )

                while True:
                    try:
                        q_data = pipeline.q.get(timeout=30)  # Add timeout
                        image_base64: str = q_data['image']
                        step: int = q_data['step']
                        total_step: int = q_data['total_step']
                        if step + 1 >= total_step:
                            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
                            break
                        elif image_base64 == 'error':
                            yield f"data: {json.dumps({'status': 'error', 'message': 'Error processing image'})}\n\n"
                            break
                        else:
                            # Send in smaller chunks
                            chunk_size = 1024*1024  # 1MB chunks
                            total_chunks = (len(image_base64) + chunk_size - 1) // chunk_size  # Calculate total chunks
                            for i in range(0, len(image_base64), chunk_size):
                                chunk = image_base64[i:i + chunk_size]
                                data = f"data: {json.dumps({'status': 'processing', 'step': step, 'total_step': total_step, 'data': [{'image': chunk, 'chunk': i//chunk_size, 'total_chunks': total_chunks}]})}\n\n"
                                yield data
                    except queue.Empty:
                        yield f"data: {json.dumps({'status': 'error', 'message': 'Timeout waiting for image'})}\n\n"
                        break
                logComponentMemory(shared_pipeline.pipeline.hf_device_map)
            except Exception as e:
                logger.exception(f"Error in generate_image: {str(e)}")
                if isinstance(e, HTTPException):
                    raise e
                elif hasattr(e, "message"):
                    raise HTTPException(status_code=500, detail=e.message + traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    except Exception as e:
        # add a log to log the exception stack and msg
        logger.exception(f"Error in generate_image: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        elif hasattr(e, "message"):
            raise HTTPException(status_code=500, detail=e.message + traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
