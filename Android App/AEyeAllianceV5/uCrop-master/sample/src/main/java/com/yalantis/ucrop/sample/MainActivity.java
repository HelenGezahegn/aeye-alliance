package com.yalantis.ucrop.sample;

import android.content.Intent;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.ImageButton;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ImageButton bt1 = (ImageButton) findViewById(R.id.leftImgButton);
        ImageButton bt2 = (ImageButton) findViewById(R.id.rightImgButton);
//        displayImg = (ImageView) findViewById(R.id.displayImg);
//        pickImgBtn = (Button) findViewById(R.id.pickImgBtn);

//        pickImgBtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                CropImage.activity().start(MainActivity.this);
//            }
//        });

        bt1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent int1 = new Intent(MainActivity.this, AboutUs.class);
                startActivity(int1);
            }
        });

        bt2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent int2 = new Intent(MainActivity.this, SampleActivity.class);
                startActivity(int2);
            }
        });

    }

    public ImageButton btn1;
    public void init1() {
        btn1 = (ImageButton)findViewById(R.id.leftImgButton);
        btn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent aboutUs = new Intent(MainActivity.this, AboutUs.class);

                startActivity(aboutUs);

            }
        });

    }


    public ImageButton btn2;
    public void init2() {
        btn2 = (ImageButton)findViewById(R.id.rightImgButton);
        btn2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent upload = new Intent(MainActivity.this, AboutUs.class);

                startActivity(upload);

            }
        });

    }


    //eyes opening
    ImageButton button1;
    ImageButton button2;


    View.OnClickListener imgButtonHandler1 = new View.OnClickListener() {
        @Override

        public void onClick(View v) {
            button1.setBackgroundResource(R.drawable.eye);
            button2.setBackgroundResource(R.drawable.eye);
            //button1.setBackgroundResource(R.drawable.eye);
            //button.setVisibility(View.GONE);
            //button.setVisibility(View.INVISIBLE);



        }
    };


    View.OnClickListener imgButtonHandler2 = new View.OnClickListener() {
        @Override

        public void onClick(View v) {
            // button2.setVisibility(View.INVISIBLE);
            button1.setBackgroundResource(R.drawable.eye);
            button2.setBackgroundResource(R.drawable.eye);

            //setting the on-click activity

            //button1.setBackgroundResource(R.drawable.eye);
            //button.setVisibility(View.GONE);
            //button.setVisibility(View.INVISIBLE);


        }
    };






}
