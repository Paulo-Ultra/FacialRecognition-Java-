package br.com.facialrecognition.reconhecimento;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGRA2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Captura {
    public static void main(String[] args) throws FrameGrabber.Exception, InterruptedException {
        KeyEvent tecla = null;
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();

        //O zero significa a web cam do notebook (Device Number), com mais câmera seguiria a ordem numeral para cada uma
        try (OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0)) {
            camera.start();

            CascadeClassifier detectorFace = new CascadeClassifier(
                    "src\\main\\java\\br\\com\\facialrecognition\\recursos\\haarcascade_frontalface_alt.xml");

            //O valor dos dois parâmetros (CanvasFrame e getGamma) são iguais o que gera o número 1 como resultado,
            //Contudo assim fica mais perfomático
            CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());

            Frame frameCapturado;
            Mat imagemColorida;
            int numeroAmostras = 25;
            int amostra = 1;
            System.out.println("Digite seu id: ");
            Scanner cadastro = new Scanner(System.in);
            int idPessoa = cadastro.nextInt();
            while ((frameCapturado = camera.grab()) != null) {
                imagemColorida = converteMat.convert(frameCapturado);

                //É necessário colocar escala de cinza para facilitar a detecção de face
                Mat imagemCinza = new Mat();
                cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
                RectVector facesDetectadas = new RectVector();

                detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0,
                        new Size(150, 150), new Size(500, 500));

                if (tecla == null) {
                    tecla = cFrame.waitKey(5);
                }

                for (int i = 0; i < facesDetectadas.size(); i++) {
                    Rect dadosFace = facesDetectadas.get(0);
                    rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));
                    Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                    resize(faceCapturada, faceCapturada, new Size(160, 160));
                    if (tecla == null) {
                        tecla = cFrame.waitKey(5);
                    }
                    if (tecla != null) {
                        if (tecla.getKeyChar() == 'q') {
                            if (amostra <= numeroAmostras) {
                                imwrite("src\\fotos\\pessoa." + idPessoa + "." + amostra + ".jpg", faceCapturada);
                                System.out.println("Foto " + amostra + " capturada\n");
                                amostra++;
                            }
                        }
                        tecla = null;
                    }
                }
                if (tecla == null) {
                    tecla = cFrame.waitKey(20);
                }

                if (cFrame.isVisible()) {
                    cFrame.showImage(frameCapturado);
                }

                if (amostra > numeroAmostras) {
                    break;
                }
            }
            cFrame.dispose();
            camera.stop();
        }
    }
}