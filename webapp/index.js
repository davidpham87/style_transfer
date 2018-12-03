/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
tf.setBackend('webgl');

const MODEL_PATH = 'http://localhost:8080/web_model/tensorflowjs_model.pb';
const WEIGHTS_PATH = 'http://localhost:8080/web_model/weights_manifest.json';

//     // tslint:disable-next-line:max-line-length
//     // 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;

let style_transfer;
const style_transfer_demo = async () => {
  status('Loading model...');

  // const manifest = await fetch(WEIGHTS_PATH);
  // const weightManifest = await manifest.json();
  // console.log(weightManifest);
  // const weightMap = await tf.io.loadWeights(weightManifest, "http://localhost:8080/web_model/");


  style_transfer = await tf.loadFrozenModel(MODEL_PATH, WEIGHTS_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  const zero_image = tf.zeros([IMAGE_SIZE, IMAGE_SIZE, 3]);

  style_transfer.predict([zero_image, zero_image]).dispose();

  status('');

  // // Make a prediction through the locally hosted cat.jpg.
  const style_element = document.getElementById('style');
  const content_element = document.getElementById('content');
  if (style_element.complete && style_element.naturalHeight !== 0) {
    predict(content_element, style_element);
    style_element.style.display = '';
    content_element.style.display = '';
  } else {
    style_element.onload = () => {
      predict({content_element, style_element});
      style_element.style.display = '';
    }
  }
  
  document.getElementById('file-container').style.display = '';
  console.log("loaded");
};

// /**
//  * Given an image element, makes a prediction through mobilenet returning the
//  * probabilities of the top K classes.
//  */
async function predict(content_element, style_element) {
  status('Predicting...');

  const startTime = performance.now();
  const generated_image = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
    const content_img = tf.fromPixels(content_element).toFloat();
    const style_img = tf.fromPixels(style_element).toFloat();

    // const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].

    // const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    // const content_batched = content_img.reshape([IMAGE_SIZE, IMAGE_SIZE, 3]);
    // const style_batched = style_img.reshape([IMAGE_SIZE, IMAGE_SIZE, 3]);

    // Make a prediction through mobilenet.
    return style_transfer.predict({"content_images_placeholder": content_img, "style_images_placeholder": style_img});
  });

  // Convert logits to probabilities and class names.
  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`);

  // Show the classes in the DOM.
  showResults(generated_image);
}

// //
// // UI
// //

function showResults(imgElement) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

style_transfer_demo();
