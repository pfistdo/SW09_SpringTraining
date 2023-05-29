package ch.pfistdo1.zhaw.springboot.training;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;

public class Models {

    public static final String MODEL_NAME = "shoeclassifier";
    public static final int BATCH_SIZE = 20;
    public static final int EPOCHS = 3;
    public static final int IMAGE_WIDTH = 224;
    public static final int IMAGE_HEIGHT = 224;
    public static final int NUM_OF_OUTPUT = 50;

    public static Model getModel() {
        // create new instance of an empty model
        Model model = Model.newInstance(MODEL_NAME);
        Block resNet50 = ResNetV1.builder() // construct the network
                .setImageShape(new Shape(3, Models.IMAGE_HEIGHT, Models.IMAGE_WIDTH))
                .setNumLayers(50)
                .setOutSize(Models.NUM_OF_OUTPUT)
                .build();
        // set the neural network to the model
        model.setBlock(resNet50);
        return model;
    }

    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }
}