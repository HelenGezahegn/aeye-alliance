package com.yalantis.ucrop.sample;

import android.content.Intent;
import android.graphics.Color;
import android.opengl.Visibility;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;

import com.felipecsl.gifimageview.library.GifImageView;

import org.apache.commons.io.IOUtils;

import java.io.IOException;
import java.io.InputStream;

import static android.view.View.INVISIBLE;


public class MainActivity extends AppCompatActivity {
    private GifImageView gifImageView;
    private ProgressBar progressBar;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final Button bt1 = (Button) findViewById(R.id.leftImgButton);
        final Button bt2 = (Button) findViewById(R.id.rightImgButton);

        gifImageView = (GifImageView) findViewById(R.id.gifImageView);
//        progressBar = (ProgressBar) findViewById(R.id.progressBar);
//        progressBar.setVisibility(progressBar.VISIBLE);

        try {
            InputStream inputStream = getAssets().open("cute_eye.gif");
            byte[] bytes = IOUtils.toByteArray(inputStream);
            gifImageView.setBytes(bytes);
            gifImageView.startAnimation();

        } catch (IOException ex) {

        }
        bt1.setOnClickListener(new View.OnClickListener() {
                                   @Override
                                   public void onClick(View v) {
                                       Intent int1 = new Intent(MainActivity.this, AboutUs.class);
                                       startActivity(int1);
                                   }
                               }
        );

        bt2.setOnClickListener(new View.OnClickListener() {
                                   @Override
                                   public void onClick(View v) {
                                       Intent int2 = new Intent(MainActivity.this, SampleActivity.class);
                                       startActivity(int2);
                                   }
                               }
        );
    }




//        bt1.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//
//
//                new CountDownTimer(300, 1000) {
//
//                    public void onTick(long millisUntilFinished) {
//                        bt1.setBackgroundResource(R.drawable.eye);
//                    }
//
//                    public void onFinish() {
//                        bt1.setBackgroundResource(0);
//
//                    }
//                }.start();
//
//            Intent int1 = new Intent(MainActivity.this, AboutUs.class);
//             startActivity(int1);
//
////              bt1.setBackgroundResource(R.drawable.eye);
////
////                Intent int1 = new Intent(MainActivity.this, AboutUs.class);
////                startActivity(int1);
//
//
//
//            }
//        });
//
//        bt2.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                new CountDownTimer(300, 1000) {
//
//                    public void onTick(long millisUntilFinished) {
//                        bt2.setBackgroundResource(R.drawable.eye);
//                    }
//
//                    public void onFinish() {
//                        bt2.setBackgroundResource(0);
//
//                    }
//                }.start();
//
//                Intent int2 = new Intent(MainActivity.this, SampleActivity.class);
//                startActivity(int2);
//
//
////                bt2.setBackgroundResource(R.drawable.eye);
////                Intent int2 = new Intent(MainActivity.this, SampleActivity.class);
////                startActivity(int2);
//            }
//        });

    //}






}
