package com.aces.pillow.camest;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

public class CamActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = CamActivity.class.getName();

    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 127);

    private static final int      thickness = 2;

    private static final int      MAXIMUM_ALLOWED_SKIPPED_FRAMES = 5;

    private static final int      MAXIMUM_ALLOWED_UNDETECTED_FRAMES = 10;

    private int                    shoulder_frame_count = 0;
    private int                    face_frame_count = 0;

    private double                width = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private Mat                    mRgba;
    private Mat                    mGray;

    private Detector               mDetector;

    private Rect                   face_frame;
    private Rect                   shoulder_frame;

    private Rect                   pre_face_frame;
    private Rect                   pre_shoulder_frame;

    private Kalman                 FaceKalman;
    private Kalman                 ShoulderKalman;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mDetector.load();

                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public CamActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_cam);

        mDetector = new Detector(this);

        mDetector.setFaceDetected(false);
        mDetector.setShoulderDetected(false);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cam_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();

        face_frame = new Rect();
        shoulder_frame = new Rect();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (!mDetector.isFaceDetected()) {
            face_frame = mDetector.FaceDetector(mGray);
            if (face_frame != null) {
                pre_face_frame = face_frame.clone();
                FaceKalman = new Kalman(new Point(face_frame.x + face_frame.width * 0.5,
                        face_frame.y + face_frame.height * 0.5));
            }
        } else {
            if (face_frame_count <= MAXIMUM_ALLOWED_SKIPPED_FRAMES) {
                if (!mDetector.isShoulderDetected()) {
                    shoulder_frame = mDetector.ShoulderDetector(mGray);
                    if (shoulder_frame != null) {
                        pre_shoulder_frame = shoulder_frame.clone();
                    }
                }
            } else if (face_frame_count > MAXIMUM_ALLOWED_SKIPPED_FRAMES
                    && face_frame_count <= MAXIMUM_ALLOWED_UNDETECTED_FRAMES) {
                face_frame = mDetector.FaceDetector(mGray);
            }
        }

        Point c, P1, P2;
        if (mDetector.isFaceDetected()) {
            if (face_frame_count == 0) {
                show_face();
                face_frame_count++;
            } else if (face_frame_count > 0 && face_frame_count <= MAXIMUM_ALLOWED_SKIPPED_FRAMES) {
                c = FaceKalman.getPrediction();
                P1 = new Point(c.x - face_frame.width / 2, c.y - face_frame.height / 2);
                P2 = new Point(c.x + face_frame.width / 2, c.y + face_frame.height / 2);
                Imgproc.rectangle(mRgba, P1, P2, FACE_RECT_COLOR, thickness);
                face_frame_count++;
            } else if (face_frame_count > MAXIMUM_ALLOWED_SKIPPED_FRAMES
                    && face_frame_count <= MAXIMUM_ALLOWED_UNDETECTED_FRAMES) {
                if (face_frame == null) {
                    face_frame = pre_face_frame.clone();
                    c = FaceKalman.getPrediction();
                    P1 = new Point(c.x - face_frame.width / 2, c.y - face_frame.height / 2);
                    P2 = new Point(c.x + face_frame.width / 2, c.y + face_frame.height / 2);
                    Imgproc.rectangle(mRgba, P1, P2, FACE_RECT_COLOR, thickness);
                    face_frame_count++;
                } else {
                    face_frame_count = 1;
                    pre_face_frame = face_frame.clone();
                    c = new Point((face_frame.tl().x + face_frame.br().x) / 2,
                            (face_frame.tl().y + face_frame.br().y) / 2 );
                    FaceKalman.correction(c);
                    show_face();
                }
            } else if (face_frame_count > MAXIMUM_ALLOWED_UNDETECTED_FRAMES) {
                face_frame_count = 0;
                mDetector.setFaceDetected(false);
                face_frame = null;
                pre_face_frame = null;
                FaceKalman = null;
            }
        }

        if (mDetector.isShoulderDetected()) {
            show_shoulder();
            shoulder_frame_count++;
            if (shoulder_frame_count > MAXIMUM_ALLOWED_UNDETECTED_FRAMES) {
                shoulder_frame_count = 0;
                mDetector.setShoulderDetected(false);
                shoulder_frame = null;
                pre_shoulder_frame = null;
            }
        }

        return mRgba;
    }

    public double getWidth() {
        return width;
    }

    private void show_face() {
        if (face_frame != null) {
            Imgproc.rectangle(mRgba, face_frame.tl(), face_frame.br(), FACE_RECT_COLOR, thickness);
        }
    }

    private void show_shoulder() {
        if (shoulder_frame != null)
            Imgproc.rectangle(mRgba, shoulder_frame.tl(), shoulder_frame.br(), FACE_RECT_COLOR, thickness);
    }

}
