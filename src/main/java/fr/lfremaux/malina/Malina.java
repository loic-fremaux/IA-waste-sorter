package fr.lfremaux.malina;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class Malina {

    private String dataPath;
    private ComputationGraph model;

    private RecordReaderDataSetIterator train;
    private RecordReaderDataSetIterator test;

    private static final String IMAGES_FOLDER = "JPEGImages";
    private static final String ANNOTATIONS_FOLDER = "Annotations";

    int width = 416;
    int height = 416;
    int nChannels = 3;
    int gridWidth = 13;
    int gridHeight = 13;

    int nClasses = 1;

    int nBoxes = 5;
    double lambdaNoObj = 0.5;
    double lambdaCoord = 5.0;
    double[][] priorBoxes = {{2, 2}, {2, 2}, {2, 2}, {2, 2}, {2, 2}};
    double detectionThreshold = 0.3;

    int batchSize = 2;
    int nEpochs = 50;
    double learningRate = 1e-3;
    double lrMomentum = 0.9;

    public Malina(String dataPath) {
        try {
            System.out.println(dataPath);
            this.dataPath = new ClassPathResource(dataPath).getFile().getPath();
            init(new File(this.dataPath, IMAGES_FOLDER));
            train();
            printResult();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void printResult() {
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("TestDetection");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) model.getOutputLayer(0);
        List<String> labels = train.getLabels();
        test.setCollectMetaData(true);
        while (test.hasNext() && frame.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI) ds.getExampleMetaData().get(0);
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            File file = new File(metadata.getURI());

            Mat mat = imageLoader.asMat(features);
            Mat convertedMat = new Mat();
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize(convertedMat, image, new Size(w, h));
            for (DetectedObject obj : objs) {
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(image, new Point(x1, y1), new Point(x2, y2), opencv_core.Scalar.RED);
                putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, opencv_core.Scalar.GREEN);
            }
            frame.setTitle(new File(metadata.getURI()).getName() + " - TestDetection");
            frame.setCanvasSize(w, h);
            frame.showImage(converter.convert(image));
            try {
                frame.waitKey();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        frame.dispose();
    }

    private void train() throws IOException {
        String modelFilename = "model_rbc.zip";
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Epoch " + i);
            train.reset();
            while (train.hasNext()) {
                System.out.println("training");
                model.fit(train.next());
            }
        }
        ModelSerializer.writeModel(model, modelFilename, true);
    }

    private void init(File path) throws IOException {
        System.out.println("Loading data...");
        final Random rng = new Random(123);
        final RandomPathFilter pathFilter = new RandomPathFilter(rng) {
            @Override
            protected boolean accept(String name) {
                name = name.replace("/" + IMAGES_FOLDER + "/", "/" + ANNOTATIONS_FOLDER + "/").replace(".jpg", ".xml");
                try {
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };
        InputSplit[] data = new FileSplit(new File(dataPath), NativeImageLoader.ALLOWED_FORMATS, rng).sample(pathFilter, 0.8, 0.2);

        InputSplit trainData = data[0];
        InputSplit testData = data[1];
        System.out.println("Data successfully loaded !");

        System.out.println(trainData.length());
        System.out.println(testData.length());

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, new VocLabelProvider(this.dataPath));

        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth,
                new VocLabelProvider(this.dataPath));
        recordReaderTest.initialize(testData);

        this.train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        this.test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
        INDArray priors = Nd4j.create(priorBoxes);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder()
                        .learningRate(learningRate)
                        .build())
                .updater(new Nesterovs.Builder()
                        .learningRate(learningRate)
                        .momentum(lrMomentum).build())
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                .build();


        this.model = new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)
                .removeVertexKeepConnections("conv2d_9")
                .addLayer("convolution2d_9",
                        new ConvolutionLayer.Builder(1, 1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + nClasses))
                                .stride(1, 1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.UNIFORM)
                                .hasBias(false)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_8")
                .addLayer("outputs", new Yolo2OutputLayer.Builder()
                                .lambbaNoObj(lambdaNoObj)
                                .lambdaCoord(lambdaCoord)
                                .boundingBoxPriors(priors)
                                .build(),
                        "convolution2d_9")
                .setOutputs("outputs")
                .build();
    }
}
