package br.com.facialrecognition.reconhecimento;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_face.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

public class Treinamento {
    public static void main(String[] args) {
        File diretorio = new File("src\\fotos");
        FilenameFilter filtroImagem = new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") || name.endsWith(".gif") || name.endsWith(".png");
            }
        };

        File[] arquivos = diretorio.listFiles(filtroImagem);
        MatVector fotos = new MatVector(arquivos.length);
        //Rótulos são os ID's
        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        for(File imagem : arquivos) {
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            //System.out.println(imagem.getName().split("\\.")[1] + "  " + imagem.getAbsolutePath());
            resize(foto, foto, new Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }

        //PCA -> Principal Component Analysis - Parâmetros do EigenFaces são os Components e EigenVectors o Padrão é 50
        //PCA -> Analisa a variação para montar uma Mean Face
        //Na documentação diz que esse número de componentes (50) é suficiente normalmente
        FaceRecognizer eigenfaces = EigenFaceRecognizer.create(50, 0);

        //LDA -> Linear Discriminant Analysis - Reduz as dimensões, e não fica focado na variação de imagens(PCA),
        // mas em maximizar a separação entre as classes e a iluminação não afetará as outras faces.
        FaceRecognizer fisherFaces = FisherFaceRecognizer.create();

        //LBPH -> Local Binary Patterns Histograms - O número decimal é usado para treinar o sistema, gerando um histograma
        //dos valores para cada face, melhor para ser usado em ambientes de muita ou pouca luz, pois consegue diferenciar melhor
        FaceRecognizer lbph = LBPHFaceRecognizer.create(2,9,9,9,1);

        eigenfaces.train(fotos, rotulos);
        eigenfaces.save("src\\main\\java\\br\\com\\facialrecognition\\recursos\\classificadorEigenFaces.yml");

        fisherFaces.train(fotos, rotulos);
        fisherFaces.save("src\\main\\java\\br\\com\\facialrecognition\\recursos\\classificadorFisherFaces.yml");

        lbph.train(fotos, rotulos);
        lbph.save("src\\main\\java\\br\\com\\facialrecognition\\recursos\\classificadorLBPH.yml");

    }
}