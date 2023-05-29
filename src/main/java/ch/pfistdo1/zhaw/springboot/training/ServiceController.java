package ch.pfistdo1.zhaw.springboot.training;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import org.springframework.web.bind.annotation.RequestParam;

@RestController
public class ServiceController {
    private Predictor<Image, Classifications> predictor;

    public ServiceController() {
        Model model = Models.getModel();
        Path modelDir = Paths.get("src/main/resources/models");
        try {
            model.load(modelDir, Models.MODEL_NAME);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .optApplySoftmax(true)
                .build();
        predictor = model.newPredictor(translator);
    }

    @GetMapping("/")
    public ModelAndView index() {
        return new ModelAndView("templates/index.html");
    }

    @GetMapping("/ping")
    public String ping() {
        return "Sentiment app is up and running!";
    }

    @PostMapping(path = "/analyze")
    public String predict(@RequestParam("image") MultipartFile image) throws Exception {
        return predict(image.getBytes()).toJson();
    }

    public Classifications predict(byte[] image) throws ModelException, TranslateException, IOException {
        InputStream is = new ByteArrayInputStream(image);
        BufferedImage bi = ImageIO.read(is);
        Image img = ImageFactory.getInstance().fromImage(bi);
        Classifications predictResult = this.predictor.predict(img);
        return predictResult;
    }
}