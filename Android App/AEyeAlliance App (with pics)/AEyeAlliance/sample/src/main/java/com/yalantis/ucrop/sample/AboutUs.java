package com.yalantis.ucrop.sample;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.text.Html;
import android.text.method.LinkMovementMethod;

public class AboutUs extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_about_us);
//       LinearLayout about_us =  (LinearLayout) findViewById(R.id.wrapper_settings);
//        about_us.setMovementMethod(new ScrollingMovementMethod());

        TextView link = (TextView) findViewById(R.id.source_code);
        String linkText = "<a href='https://github.com/HelenG123/aeye-alliance'> Source Code </a>";
        link.setText(Html.fromHtml(linkText));
        link.setMovementMethod(LinkMovementMethod.getInstance());


    }


}
