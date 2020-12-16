from torchvision import models
import torch
import numpy as np
from skimage.transform import resize
import cv2

#Define prepocessing steps
def preprocess_image(img_path):
    cnn_input_size = (224,224)
    mean=[0.485, 0.456, 0.406]
    std =[0.229,0.224,0.225]

    #read input image
    input_img = cv2.imread(img_path)
    #do transformations
    input_img = resize(input_img, cnn_input_size, mode = 'reflect', anti_aliasing = True).astype(np.float32)
    input_img -= mean
    input_img /= std
    # Convert HWC -> CHW
    input_img = input_img.transpose(2, 0, 1)
    # Load to GPU tensor
    input_img = torch.tensor(input_img)
    #apply batch data
    batch_data = torch.unsqueeze(input_img, 0)
    return batch_data

#Define postprocessing steps
def postprocess_output(output_data):
    # Get the labels:
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    # Calculate human-readable value with softmax function
    confidences = torch.nn.functional.softmax(output_data, dim = 1)[0] * 100 #in %
    # Find top predicted classes
    _, indices = torch.sort(output_data, descending = True)
    i = 0
    # Print those with high probability
    while confidences[indices[0][i]] > 0.5:
        label_idx = indices[0][i]
        print("Label: ", labels[label_idx], ", confidence:",confidences[label_idx].item(),"%, index:",label_idx.item(), )
        i += 1

#Main loop
def main():
    #Load image, do preprocessing on CPU, then past on to GPU tensor
    image_path = "./turkish_coffee.jpg"
    img = preprocess_image(image_path).cuda()

    #Set up model (py torch)
    model = models.resnet50(pretrained=True)
    model.eval()
    model.cuda()

    #Set up loggers for time measurement
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)

    #Warming up GPU
    for _ in range(10):
        _ = model(img)

    #Measuring average time
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            output = model(img)
            ender.record()
            #Syncrhonize GPU
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_infer_time = np.sum(timings) / repetitions
    std_infer_time = np.std(timings)
    print("Infer time: ", mean_infer_time, "+-", std_infer_time)

    #Post process
    postprocess_output(output)

if __name__ == "__main__":
    main()

