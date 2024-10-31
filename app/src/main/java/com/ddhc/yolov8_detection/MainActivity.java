// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.ddhc.yolov8_detection;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.PixelCopy;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.widget.AppCompatSeekBar;

import com.hjq.permissions.OnPermissionCallback;
import com.hjq.permissions.Permission;
import com.hjq.permissions.XXPermissions;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;
    private static final String TAG = "ncnn";
    private Yolov8Ncnn yolov8ncnn = new Yolov8Ncnn();
    private int facing = 1;//默认后置摄像头
    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;

    public void setFullscreen(boolean isShowStatusBar, boolean isShowNavigationBar) {
        int uiOptions = View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        if (!isShowStatusBar) {
            uiOptions |= View.SYSTEM_UI_FLAG_FULLSCREEN;
        }
        if (!isShowNavigationBar) {
            uiOptions |= View.SYSTEM_UI_FLAG_HIDE_NAVIGATION;
        }
        getWindow().getDecorView().setSystemUiVisibility(uiOptions);
        setNavigationStatusColor(Color.TRANSPARENT);
    }

    public void setNavigationStatusColor(int color) {
        if (Build.VERSION.SDK_INT >= 21) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS);
            getWindow().setNavigationBarColor(color);
            getWindow().setStatusBarColor(color);
        }
    }

    FrameLayout root;
    ImageView pic;
    FlowLayout cls;
    TextView rect;
    ArrayList<TextView> textViews = new ArrayList<>();
    float sX;
    float sY;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.main);

        //全屏
        setFullscreen(true, true);

        DisplayMetrics dm = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getRealMetrics(dm);
        int width = dm.widthPixels;
        int height = dm.heightPixels;
        Log.e(TAG, "width: " + width + ",height:" + height); //720,1560

        root = findViewById(R.id.root);
        pic = findViewById(R.id.pic);
        cls = findViewById(R.id.cls);


        cls.setOnChildClickListener(new FlowLayout.OnChildClickListener() {
            @Override
            public void onChildClick(String cls, int color) {
                if (rect != null) {
                    rect.setText(cls);
                    rect.setBackgroundColor(color);
                    rect.setFocusable(true);
                    rect.setTag(true);
                    rect.setOnLongClickListener(v1 -> {
                        root.removeView(v1);
                        return true;
                    });
                }
            }
        });

        int warn_color = Color.argb(70, 255, 0, 0);
        int ok_color = Color.argb(70, 0, 255, 0);
        pic.setOnTouchListener((v, event) -> {
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    if (rect != null) {
                        if (rect.getText().toString().equals("")) {
                            root.removeView(rect);
                            rect = null;
                        }
                    }
                    rect = new TextView(v.getContext());
                    rect.setFocusable(false);
                    rect.setBackgroundColor(warn_color);
                    sX = event.getX();
                    sY = event.getY();
                    rect.setX(sX);
                    rect.setY(sY);
                    root.addView(rect, 0, 0);
                    return true;
                case MotionEvent.ACTION_MOVE:
                    float x = event.getX();
                    float y = event.getY();
                    rect.setX(Math.min(x, sX));
                    rect.setY(Math.min(y, sY));
                    int w = rect.getLayoutParams().width = (int) Math.abs(x - sX);
                    int h = rect.getLayoutParams().height = (int) Math.abs(y - sY);
                    if (w > 100 && h > 100) {
                        rect.setBackgroundColor(ok_color);
                    } else {
                        rect.setBackgroundColor(warn_color);
                    }
                    rect.requestLayout();
                    break;
                case MotionEvent.ACTION_UP:
                    if (rect.getWidth() >= 100 && rect.getHeight() >= 100) {
                        cls.setVisibility(View.VISIBLE);
                    } else {
                        root.removeView(rect);
                        rect = null;
                        cls.setVisibility(View.GONE);
                        Toast.makeText(MainActivity.this, "绘制区域过小...", Toast.LENGTH_SHORT).show();
                    }
                    break;
            }
            return false;
        });

        AppCompatSeekBar seekBar = findViewById(R.id.seekBar);

        TextView seek = findViewById(R.id.seek);
        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                System.out.println("ncnn  seekbar " + progress);
                seek.setText("置信度:" + progress);
                yolov8ncnn.setProb(progress / 100.f);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });


        //屏幕保持开启
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);

        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                yolov8ncnn.closeCamera();

                yolov8ncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_model) {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        reload();
    }


    private void reload() {
        Log.i(TAG, "reload: " + current_model);
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init) {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }
    }


    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        yolov8ncnn.setOutputWindow(holder.getSurface());

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    @Override
    public void onResume() {
        super.onResume();
        Button b;
        XXPermissions.with(this)
                .permission(Permission.CAMERA)
                .request(new OnPermissionCallback() {

                    @Override
                    public void onGranted(@NonNull List<String> permissions, boolean allGranted) {
                        yolov8ncnn.openCamera(facing);
                    }

                    @Override
                    public void onDenied(@NonNull List<String> permissions, boolean doNotAskAgain) {

                    }
                });
//        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
//        {
//            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
//        }


    }

    @Override
    public void onPause() {
        super.onPause();

        yolov8ncnn.closeCamera();
    }

    Bitmap mScreenBitmap;

    /**
     * 数据集
     *
     * @param view
     */
    public void addData(View view) {
        //需要截取的长和宽
        int outWidth = cameraView.getWidth();
        int outHeight = cameraView.getHeight();

        mScreenBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888);
        PixelCopy.request(cameraView, mScreenBitmap, copyResult -> {
            if (PixelCopy.SUCCESS == copyResult) {
                root.setVisibility(View.VISIBLE);
                pic.setImageBitmap(mScreenBitmap);
            } else {
                Toast.makeText(MainActivity.this, "截图失败...", Toast.LENGTH_SHORT).show();
            }
        }, new Handler());
    }

    public void bcakToMain(View view) {
        Log.i(TAG, root.getChildCount() + "");

        if (root.getChildCount() > 3) {
            //1.存储原图片
            String base64pic = BitmapUtil.bitmapToBase64(mScreenBitmap);
            Log.i(TAG, "ori: " + mScreenBitmap.getWidth() + " " + mScreenBitmap.getHeight());
            while (root.getChildCount() > 3) {
                //2.存储标注数据
                TextView childAt = (TextView) root.getChildAt(root.getChildCount() - 1);
                String label = childAt.getText().toString();
                if (!label.equals("")) {
                    Log.i(TAG, "cls: " + label + " " + childAt.getX() + " " + childAt.getY() + " " + childAt.getWidth() + " " + childAt.getHeight());
                }
                //3.删除标注数据图层
                root.removeView(childAt);
            }
        }
        //4.隐藏素材标注页面
        root.setVisibility(View.GONE);
        cls.setVisibility(View.GONE);
    }
}
