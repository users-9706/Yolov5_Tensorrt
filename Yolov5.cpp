
#include <windows.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
std::vector<std::string> read_class_names(std::string path_name)
{
	std::vector<std::string> class_names;
	std::ifstream infile;
	infile.open(path_name.data());   
	assert(infile.is_open());   
	std::string str;
	while (getline(infile, str)) {
		class_names.push_back(str);
		str.clear();
	}
	infile.close();
	return class_names;
}
class Logger : public nvinfer1::ILogger
{
	void log(Severity severity, const char* message)  noexcept
	{
		if (severity != Severity::kINFO)
			std::cout << message << std::endl;
	}
} gLogger;
void onnx_to_engine(std::string onnx_file_path, std::string engine_file_path, std::string type) {
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	parser->parseFromFile(onnx_file_path.c_str(), 2);
	for (int i = 0; i < parser->getNbErrors(); ++i) 
	{
		std::cout << "load error: " << parser->getError(i)->desc() << std::endl;
	}
	printf("tensorRT load mask onnx model successfully!!!...\n");
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	config->setMaxWorkspaceSize(1 << 30);
	if (type == "fp16") {
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else {
		std::cout << "WARNING: FP32 is used by default." << std::endl;
	}
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "try to save engine file now~~~" << std::endl;
	std::ofstream file_ptr(engine_file_path, std::ios::binary);
	if (!file_ptr) {
		std::cerr << "could not open plan output file" << std::endl;
		return;
	}
	nvinfer1::IHostMemory* model_stream = engine->serialize();
	file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
	model_stream->destroy();
	engine->destroy();
	network->destroy();
	parser->destroy();
	std::cout << "convert onnx model to TensorRT engine model successfully!" << std::endl;
}
int main() 
{
	const char* model_path_onnx = "yolov5s.onnx";
	const char* model_path_engine = "yolov5s.engine";
	const char* image_path = "bus.jpg";
	std::string label_path = "coco.names";
	const char* input_node_name = "images";
	const char* output_node_name = "output0";
	int num_ionode = 2;
	std::vector<std::string> class_names;
	float factor;
	std::ifstream f(model_path_engine);
	bool engine_file_exist = f.good();
	std::ifstream file_ptr(model_path_engine, std::ios::binary);
	if (engine_file_exist) {
		size_t size = 0;
		file_ptr.seekg(0, file_ptr.end);	
		size = file_ptr.tellg();	
		file_ptr.seekg(0, file_ptr.beg);	
		char* model_stream = new char[size];
		file_ptr.read(model_stream, size);
		file_ptr.close();
		Logger logger;
		nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
		nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(model_stream, size);
		nvinfer1::IExecutionContext* context = engine->createExecutionContext();
		delete[] model_stream;
		void** data_buffer = new void* [num_ionode];
		int input_node_index = engine->getBindingIndex(input_node_name);
		nvinfer1::Dims input_node_dim = engine->getBindingDimensions(input_node_index);
		size_t input_data_length = input_node_dim.d[1] * input_node_dim.d[2] * input_node_dim.d[3];
		cudaMalloc(&(data_buffer[input_node_index]), input_data_length * sizeof(float));
		int output_node_index = engine->getBindingIndex(output_node_name);
		nvinfer1::Dims output_node_dim = engine->getBindingDimensions(output_node_index);
		size_t output_data_length = output_node_dim.d[1] * output_node_dim.d[2];
		cudaMalloc(&(data_buffer[output_node_index]), output_data_length * sizeof(float));
		cv::Mat image = cv::imread(image_path);
		int max_side_length = std::max(image.cols, image.rows);
		cv::Mat max_image = cv::Mat::zeros(cv::Size(max_side_length, max_side_length), CV_8UC3);
		cv::Rect roi(0, 0, image.cols, image.rows);
		image.copyTo(max_image(roi));
		cv::Size input_node_shape(input_node_dim.d[2], input_node_dim.d[3]);
		int64 start = cv::getTickCount();
		cv::Mat BN_image = cv::dnn::blobFromImage(max_image, 1 / 255.0, input_node_shape, cv::Scalar(0, 0, 0), true, false);
		cudaStream_t stream;
		cudaStreamCreate(&stream);
		std::vector<float> input_data(input_data_length);
		memcpy(input_data.data(), BN_image.ptr<float>(), input_data_length * sizeof(float));
		cudaMemcpyAsync(data_buffer[input_node_index], input_data.data(), input_data_length * sizeof(float), cudaMemcpyHostToDevice, stream);
		context->enqueueV2(data_buffer, stream, nullptr);
		float* result_array = new float[output_data_length];
		cudaMemcpyAsync(result_array, data_buffer[output_node_index], output_data_length * sizeof(float), cudaMemcpyDeviceToHost, stream);
		factor = max_side_length / (float)input_node_dim.d[2];
		class_names = read_class_names(label_path);
		cv::Mat det_output = cv::Mat(25200, 85, CV_32F, result_array);
		std::vector<cv::Rect> position_boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::cout << det_output.rows << std::endl;
		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < 0.2) {
				continue;
			}
			std::cout << "confidence" << confidence << std::endl;
			cv::Mat classes_scores = det_output.row(i).colRange(5, 85);
			cv::Point classIdPoint;
			double score;
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
			if (score > 0.25)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>((cx - 0.5 * ow) * factor);
				int y = static_cast<int>((cy - 0.5 * oh) * factor);
				int width = static_cast<int>(ow * factor);
				int height = static_cast<int>(oh * factor);
				cv::Rect box;
				box.x = x;
				box.y = y;
				box.width = width;
				box.height = height;
				position_boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(position_boxes, confidences, 0.25, 0.45, indexes);
		for (size_t i = 0; i < indexes.size(); i++) {
			int index = indexes[i];
			int idx = classIds[index];
			cv::rectangle(image, position_boxes[index], cv::Scalar(0, 0, 255), 2, 8);
			cv::rectangle(image, cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y - 20),
				cv::Point(position_boxes[index].br().x, position_boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(image, class_names[idx], cv::Point(position_boxes[index].tl().x, position_boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		}
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	   	cv::putText(image, cv::format("FPS: %.2f", 1 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("C++ + Tensorrt + Yolov5 推理结果", image);
		cv::waitKey(0);
	}
	else
	{
		onnx_to_engine(model_path_onnx, model_path_engine, "fp16");
	}
	return 0;
}
