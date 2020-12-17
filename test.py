from torchvision import models
import torch
import numpy as np
from skimage.transform import resize
import cv2

#TensorRT
import onnx
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt 

#FLAG
BASE_WORKFLOW = False
TENSORRT_WORKFLOW = True
ONNX_FILE_PATH = "resnet50.onnx"
#TRT Logger
TRT_LOGGER = trt.Logger()

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
    print(input_img.shape)
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

#Define function to build engine
def build_engine(onnx_file_path):
    #Init tensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    #Parse ONNX model
    result = parser.parse_from_file(ONNX_FILE_PATH)
    print("Completed parsing of ONNX file, status ==",result)

    #Some config:

     
    #Generate the TensorRT engine optimized for Jetson Xavier
    print("Building an engine...")
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context

#Main loop
def main():
    #Load image, do preprocessing on CPU, then past on to GPU tensor
    image_path = "./turkish_coffee.jpg"
    input = preprocess_image(image_path).cuda()

    #Set up model (py torch)
    model = models.resnet50(pretrained=True).cuda()
    model.eval()

    # #Set up loggers for time measurement
    # repetitions = 10
    # timings = np.zeros((repetitions, 1))
    # starter, ender = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)

    # #Warming up GPU
    # for _ in range(10):
    #     _ = model(img)
    output = model(input)
    # #Measuring average time
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         output = model(img)
    #         ender.record()
    #         #Syncrhonize GPU
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time

    # mean_infer_time = np.sum(timings) / repetitions
    # std_infer_time = np.std(timings)
    # print("Infer time: ", mean_infer_time, "+-", std_infer_time)

    #Post process
    postprocess_output(output)

    #Convert to ONNX
    torch.onnx.export(model, input, ONNX_FILE_PATH, export_params=True)

if __name__ == "__main__":
    if BASE_WORKFLOW:
        main()

    if TENSORRT_WORKFLOW:
        #Check if ONNX model is export correctly
        onnx_temp = onnx.load(ONNX_FILE_PATH)
        print(type(onnx_temp))

        # Check that the IR is well formed
        onnx.checker.check_model(onnx_temp)
        print("Checked")

        engine, context = build_engine(ONNX_FILE_PATH)

    #Get size of input output and allocate memory
    for binding in engine:
        if engine.binding_is_input(binding):
            #Only one input in that list
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
            device_input = cuda.mem_alloc(input_size)
        else:
            #Only one output
            output_shape = engine.get_binding_shape(binding)
            #Create page locked memory buffer 
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    
    #Create a stream in which to copy inputs and outputs and run inference
    stream = cuda.Stream()
        #starter = cuda.Event(cuda.event_flags.BLOCKING_SYNC)
        #ender = cuda.Event(cuda.event_flags.BLOCKING_SYNC)
        starter, ender = cuda.Event(), cuda.Event()
        repetitions = 100
        timings = np.zeros((repetitions, 1))

    #Preprocess data
    host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order = 'C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

        # Warming up
        for _ in range (10):
            context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        
        stream.synchronize()

        #Real stuffs for inference measurement
        for rep in range(repetitions):
            starter.record(stream)
            context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
            ender.record(stream)
            ender.synchronize()
            #print("Time:",starter.time_till(ender))
            timings[rep] = starter.time_till(ender)

        cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

        mean_infer_time = np.sum(timings) / repetitions
        std_infer_time = np.std(timings)
        print("Infer time: ", mean_infer_time, "+-", std_infer_time)

        output_data = torch.Tensor(host_output)

    #Postprocess results
        output_data = output_data.reshape(output_shape[0],output_shape[1])
    postprocess_output(output_data)


if __name__ == "__main__":
    main()
        
