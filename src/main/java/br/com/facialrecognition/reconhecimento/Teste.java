package br.com.facialrecognition.reconhecimento;

import org.bytedeco.opencv.opencv_face.*; 

public class Teste {
    
    public static void main(String args[]){

        FaceRecognizer r = EigenFaceRecognizer.create();
    }
}