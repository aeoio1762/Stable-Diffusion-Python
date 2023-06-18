from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

app = Flask(__name__)

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)
@app.route('/api/resize-image', methods=['POST'])
def resize_image():
    try:
        prompt = request.form['prompt']
        image = pipe(prompt).images[0]
        new_width = 3216
        new_height = 4832

        resized_image = image.resize((new_width, new_height))


        background_image = request.files['background-image']
        background_image = Image.open(background_image)

        background_width = int(new_width * 0.5)  
        background_height = int(new_height * 0.5)  
        background_image = background_image.resize((background_width, background_height))

    
        x_offset = (new_width - background_width) // 2
        y_offset = (new_height - background_height) // 2

     
        resized_image.paste(background_image, (x_offset, y_offset), mask=background_image)

        resized_image_path = "resized_image.png"
        resized_image.save(resized_image_path)

        return jsonify({"status": "success", "resized_image_path": resized_image_path})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


