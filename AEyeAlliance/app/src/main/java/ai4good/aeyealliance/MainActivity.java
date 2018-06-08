package ai4good.aeyealliance;

import android.app.ActionBar;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.ImageButton;
import android.app.Activity;
import android.view.ViewAnimationUtils;
import android.graphics.drawable.AnimationDrawable;


public class MainActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        //added the right image button activity
        button1= (ImageButton)findViewById(R.id.leftImgButton);
        button1.setOnClickListener(imgButtonHandler1);

        button2 = (ImageButton)findViewById(R.id.rightImgButton);
        button2.setOnClickListener(imgButtonHandler2);

    }
//    ActionBar actionBar = getActionBar();
//
//    actionBar.hide();

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

//    @Override
//    public void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//
//    }


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
