# main.py content for stable-diffusion-worker

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# --- Import your Stable Diffusion model/pipeline here ---
# Example: from diffusers import StableDiffusionPipeline
# Example: import torch

# Initialize FastAPI app
app = FastAPI(
    title="Stable Diffusion Worker",
    description="Generates images using Stable Diffusion on GPU."
)

# Placeholder for your model. This will be loaded when the app starts.
# It's best to load models at the global scope or within an async lifespan context
# to ensure they are loaded only once when the FastAPI app starts.
stable_diffusion_model = None

@app.on_event("startup")
async def startup_event():
    """
    Load your Stable Diffusion model to GPU when the FastAPI app starts.
    This ensures the model is ready and loaded only once.
    """
    global stable_diffusion_model
    print("Loading Stable Diffusion model...")
    try:
        # --- REPLACE THIS WITH YOUR ACTUAL MODEL LOADING CODE ---
        # This is an example using diffusers. You'll need to install 'diffusers' and 'torch' in requirements.txt
        from diffusers import StableDiffusionPipeline
        import torch

        # Load the model. You can choose different models from Hugging Face.
        # "runwayml/stable-diffusion-v1-5" is a common starting point.
        # torch_dtype=torch.float16 uses half-precision for faster GPU performance and less memory.
        stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        # THIS IS THE CRITICAL PART: Move the model to the GPU!
        stable_diffusion_model = stable_diffusion_model.to("cuda")

        print("Stable Diffusion model loaded successfully to GPU.")
    except Exception as e:
        print(f"Failed to load Stable Diffusion model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources if necessary when the app shuts down.
    """
    print("Shutting down Stable Diffusion worker.")
    global stable_diffusion_model
    # You might want to unset the model or release GPU memory here if applicable
    stable_diffusion_model = None

class ImageGenerateRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Simple health check to ensure the worker is running and responsive.
    """
    if stable_diffusion_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "worker running", "model_status": "loaded"}

# --- Image Generation Endpoint ---
@app.post("/generate-image")
async def generate_image(request_data: ImageGenerateRequest, request: Request):
    """
    Receives a prompt and generates an image using Stable Diffusion.
    """
    # Basic API Key Authentication (important for security!)
    # Get the API key from environment variable
    expected_api_key = os.getenv("API_SECRET_KEY")
    if not expected_api_key:
        # In a real setup, you should ensure this env var is always set
        raise HTTPException(status_code=500, detail="API_SECRET_KEY not configured on server.")

    # Get the X-API-KEY header from the incoming request
    incoming_api_key = request.headers.get("X-API-KEY")

    if not incoming_api_key or incoming_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key.")

    if stable_diffusion_model is None:
        raise HTTPException(status_code=503, detail="Stable Diffusion model is not loaded.")

    print(f"Received request to generate image for prompt: '{request_data.prompt}'")

    try:
        # --- REPLACE THIS WITH YOUR ACTUAL IMAGE GENERATION LOGIC ---
        # This is an example using the loaded diffusers pipeline:
        image = stable_diffusion_model(
            request_data.prompt,
            width=request_data.width,
            height=request_data.height,
            num_inference_steps=request_data.num_inference_steps,
            guidance_scale=request_data.guidance_scale
        ).images[0] # .images[0] gets the first (and usually only) image from the output

        # Now, convert the generated PIL Image object to a Base64 string
        import io
        import base64
        # Pillow (PIL) is typically used for image manipulation
        # Make sure 'Pillow' is in your requirements.txt!
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG") # Save the image to a byte buffer as PNG
        base64_image = base64.b64encode(output_buffer.getvalue()).decode("utf-8") # Encode to Base64

        print("Image generation complete.")
        return {"image_base64": base64_image}

    except Exception as e:
        print(f"Error during image generation: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# This block allows you to run the FastAPI app directly for local testing
if __name__ == "__main__":
    # Ensure API_SECRET_KEY is set for local testing
    os.environ["API_SECRET_KEY"] = "your_strong_local_test_key" # IMPORTANT: CHANGE THIS FOR LOCAL TESTING
    uvicorn.run(app, host="0.0.0.0", port=8000)
