
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
def onnx_to_engine(onnx_file_path, engine_file_path, precision_type=None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if precision_type == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print('WARNING: FP32 is used by default.')
    profile = builder.create_optimization_profile()
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
def readClassesNames(file_path):
    with open(file_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
conf_thresold = 0.25
iou_threshold = 0.45
score_thresold = 0.25
classes_names = 'coco.names'
onnx_path = 'yolov5s.onnx'
engine_path = 'yolov5s.engine'
classes = readClassesNames(classes_names)
image = cv2.imread("bus.jpg")
image_height, image_width = image.shape[:2]
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
if os.path.exists(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    input_shape = engine.get_binding_shape(engine.get_binding_index('images'))
    input_width, input_height = input_shape[2:]
    resized = cv2.resize(image, (input_width, input_height))
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    start_time = cv2.getTickCount()
    inputs_alloc_buf = []
    outputs_alloc_buf = []
    bindings_alloc_buf = []
    stream_alloc_buf = cuda.Stream()
    context = engine.create_execution_context()
    data_type = []
    for binding in engine:
        if engine.binding_is_input(binding):
            size = input_tensor.shape[0] * input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            data_type.append(dtype)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings_alloc_buf.append(int(device_mem))
            inputs_alloc_buf.append(HostDeviceMem(host_mem, device_mem))
        else:
            size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, data_type[0])
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings_alloc_buf.append(int(device_mem))
            outputs_alloc_buf.append(HostDeviceMem(host_mem, device_mem))
    inputs_alloc_buf[0].host = input_tensor.reshape(-1)
    for inp in inputs_alloc_buf:
        cuda.memcpy_htod_async(inp.device, inp.host, stream_alloc_buf)
    context.set_binding_shape(0, input_tensor.shape)
    context.execute_async(batch_size=1, bindings=bindings_alloc_buf, stream_handle=stream_alloc_buf.handle)
    for out in outputs_alloc_buf:
        cuda.memcpy_dtoh_async(out.host, out.device, stream_alloc_buf)
    stream_alloc_buf.synchronize()
    net_output = [out.host for out in outputs_alloc_buf]
    predictions = net_output[0].reshape(25200, 85)
    scores = np.max(predictions[:, 4:5], axis=1)
    predictions = predictions[scores > score_thresold, :]
    scores = scores[scores > score_thresold]
    class_ids = np.argmax(predictions[:, 5:], axis=1)
    boxes = predictions[:, :4]
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thresold, nms_threshold=iou_threshold)
    detections = []
    def xywh2xyxy(x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        cls = classes[cls_id]
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2, 8)
        cv2.rectangle(image, (bbox[0], (bbox[1] - 20)), (bbox[2], bbox[1]), (0, 255, 255), -1)
        cv2.putText(image, f'{cls}', (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 2, [225, 0, 0], thickness=2)
    end_time = cv2.getTickCount()
    t = (end_time - start_time) / cv2.getTickFrequency()
    fps = 1 / t
    print(f"EStimated FPS: {fps:.2f}")
    cv2.putText(image, 'FPS: {:.2f}'.format(fps), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, [225, 0, 0], 2, 8);
    cv2.imshow("Python + Tensorrt + Yolov5 推理结果", image)
    cv2.waitKey(0)
else:
    onnx_to_engine(onnx_path, engine_path, 'fp16')
