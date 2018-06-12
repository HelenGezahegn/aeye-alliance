package ai4good.aeyealliance;

import android.app.ActionBar;
import android.net.Uri;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageButton;
import android.webkit.WebView;


import android.app.Activity;
import android.view.ViewAnimationUtils;
import android.graphics.drawable.AnimationDrawable;
import android.content.Intent;
import android.widget.ImageView;
import android.widget.Toast;

//import com.theartofdev.edmodo.cropper.CropImage;

import java.io.File;


public class MainActivity extends AppCompatActivity {

    ImageView displayImg;
    Button pickImgBtn;


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
                Intent int2 = new Intent(MainActivity.this, Upload.class);
                startActivity(int2);
            }
        });

    }

//    @Override
//    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//        super.onActivityResult(requestCode, resultCode, data);
//        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
//            CropImage.ActivityResult result = CropImage.getActivityResult(data);
//            if (resultCode == RESULT_OK) {
//                Uri resultUri = result.getUri();
//
//                displayImg.setImageURI(resultUri);
//
//            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
//                Exception error = result.getError();
//                Toast.makeText(this, ""+error, Toast.LENGTH_SHORT).show();
//            }
//        }
//    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

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
