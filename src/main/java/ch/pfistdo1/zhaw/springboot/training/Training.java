package ch.pfistdo1.zhaw.springboot.training;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * In training, multiple passes (or epochs) are made over the training data
 * trying to find patterns
 * and trends in the data, which are then stored in the model. During the
 * process, the model is
 * evaluated for accuracy using the validation data. The model is updated with
 * findings over each
 * epoch, which improves the accuracy of the model.
 */
public final class Training {

    public static final String MODEL_NAME = "shoeclassifier";
    public static final int BATCH_SIZE = 20;
    public static final int EPOCHS = 3;
    public static final int IMAGE_WIDTH = 224;
    public static final int IMAGE_HEIGHT = 224;
    public static final int NUM_OF_OUTPUT = 50;

    public static void main(String[] args) throws IOException, TranslateException {

        ImageFolder dataset = initDataset("src/main/resources/static/data/ut-zap50k-images-square/");
        RandomAccessDataset[] datasets = dataset.randomSplit(8, 2);

        Loss loss = Loss.softmaxCrossEntropyLoss();
        TrainingConfig config = setupTrainingConfig(loss);

        try (Model model = getModel();
                Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());
            Shape inputShape = new Shape(1, 3, IMAGE_HEIGHT, IMAGE_HEIGHT);
            trainer.initialize(inputShape);
            EasyTrain.fit(trainer, EPOCHS, datasets[0], datasets[1]);
            TrainingResult result = trainer.getTrainingResult();
            model.setProperty("Epoch", String.valueOf(EPOCHS));
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
            model.save(Path.of("src/main/resources"), MODEL_NAME);
        }

    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
                // retrieve the data
                .setRepositoryPath(Paths.get(datasetRoot))
                .optMaxDepth(10)
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                // random sampling; don't process the data in order
                .setSampling(BATCH_SIZE, true)
                .build();
        dataset.prepare();
        return dataset;
    }

    public static Model getModel() {
        // create new instance of an empty model
        Model model = Model.newInstance(MODEL_NAME);
        Block resNet50 = ResNetV1.builder() // construct the network
                .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
                .setNumLayers(50)
                .setOutSize(NUM_OF_OUTPUT)
                .build();
        // set the neural network to the model
        model.setBlock(resNet50);
        return model;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}