package com.yalantis.ucrop.sample;

import android.Manifest;
import android.annotation.TargetApi;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.app.NotificationCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.ActionBar;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

import com.yalantis.ucrop.view.UCropView;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Calendar;
import java.util.List;

import static android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION;
import static android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Integrated by AEyeAlliance
 */
public class ResultActivity extends BaseActivity {

    private static final String TAG = "ResultActivity";
    private static final String CHANNEL_ID = "3000";
    private static final int DOWNLOAD_NOTIFICATION_ID_DONE = 911;

    private static final String MODEL_FILE = "file:///assets/tf_model.pb";
    private static final String INPUT_NODE = "INPUT";

//    private static final String OUTPUT_NODE = "OUTPUT";
//    private String[] outputNames;
    String[] outputNames = {"O", "U", "T", "P", "U", "T"};


    private static final int INPUT_SIZE = 28;

    //create a TensorFLowInferenceInterface instance
    private TensorFlowInferenceInterface inferenceInterface;


    static {
        System.loadLibrary("tensorflow_inference");
    }


    public static void startWithUri(@NonNull Context context, @NonNull Uri uri) {
        Intent intent = new Intent(context, ResultActivity.class);
        intent.setData(uri);
        context.startActivity(intent);
    }

    private int[] intValues = new int[28*28];
    private float[] floatValues = new float[28* 28 * 3];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);
        Uri uri = getIntent().getData();
        if (uri != null) {
            try {
                UCropView uCropView = findViewById(R.id.ucrop);
                uCropView.getCropImageView().setImageUri(uri, null);
                uCropView.getOverlayView().setShowCropFrame(false);
                uCropView.getOverlayView().setShowCropGrid(false);
                uCropView.getOverlayView().setDimmedColor(Color.TRANSPARENT);

                // Convert image into bitmap
               // Bitmap image = uCropView.getCropImageView().getViewBitmap();
                Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

                for (int i = 0; i < intValues.length; ++i) {
                    floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
                    floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
                    floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
                }

                inferenceInterface.feed(INPUT_NODE, floatValues, 1, 28, 28, 3);
                inferenceInterface.run(outputNames, false);

                // Stopped here
                final int gridWidth = image.getWidth();
                final int gridHeight = image.getHeight();
                final float[] output = new float[28*28*3*1];
                inferenceInterface.fetch(outputNames[0], output);

                //TextView newTxtView = (TextView) findViewById(R.id.txtViewResult);
//                newTxtView.setText("hello");

                Resources res = getResources();
                String musicType = "Rock";
               // newTxtView.setText(getString(R.string.format_crop_result_d_d, options.outWidth, options.outHeight));

               // newTxtView.setText(String.format(res.getString(R.string.welcome_messages), "help", 2));
                //((TextView) findViewById(R.id.txtViewResult)).setText("Hello");
//                txtView.append("Hello");
//                TextView newTxtView = (TextView) findViewById(R.id.txtViewResult)
                //txtView.setText(Float.toString(output[0]));


            } catch (Exception e) {
                Log.e(TAG, "setImageUri", e);
                Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(new File(getIntent().getData().getPath()).getAbsolutePath(), options);

        setSupportActionBar((Toolbar) findViewById(R.id.toolbar));
        final ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setTitle(getString(R.string.format_crop_result_d_d, options.outWidth, options.outHeight));
        }
        TextView newTxtView = (TextView) findViewById(R.id.txtViewResult);
        newTxtView.setText(getString(R.string.bonnie_boy, 30000, 2));



    }

    @Override
    public boolean onCreateOptionsMenu(final Menu menu) {
        getMenuInflater().inflate(R.menu.menu_result, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == R.id.menu_download) {
            saveCroppedImage();
        } else if (item.getItemId() == android.R.id.home) {
            onBackPressed();
        }
        return super.onOptionsItemSelected(item);
    }


    /**
     * Callback received when a permissions request has been completed.
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case REQUEST_STORAGE_WRITE_ACCESS_PERMISSION:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    saveCroppedImage();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private void saveCroppedImage() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    getString(R.string.permission_write_storage_rationale),
                    REQUEST_STORAGE_WRITE_ACCESS_PERMISSION);
        } else {
            Uri imageUri = getIntent().getData();
            if (imageUri != null && imageUri.getScheme().equals("file")) {
                try {
                    copyFileToDownloads(getIntent().getData());
                } catch (Exception e) {
                    Toast.makeText(ResultActivity.this, e.getMessage(), Toast.LENGTH_SHORT).show();
                    Log.e(TAG, imageUri.toString(), e);
                }
            } else {
                Toast.makeText(ResultActivity.this, getString(R.string.toast_unexpected_error), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void copyFileToDownloads(Uri croppedFileUri) throws Exception {
        String downloadsDirectoryPath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();
        String filename = String.format("%d_%s", Calendar.getInstance().getTimeInMillis(), croppedFileUri.getLastPathSegment());

        File saveFile = new File(downloadsDirectoryPath, filename);

        FileInputStream inStream = new FileInputStream(new File(croppedFileUri.getPath()));
        FileOutputStream outStream = new FileOutputStream(saveFile);
        FileChannel inChannel = inStream.getChannel();
        FileChannel outChannel = outStream.getChannel();
        inChannel.transferTo(0, inChannel.size(), outChannel);
        inStream.close();
        outStream.close();

        showNotification(saveFile);
        Toast.makeText(this, R.string.notification_image_saved, Toast.LENGTH_SHORT).show();
        finish();

    }

    private void showNotification(@NonNull File file) throws FileNotFoundException {
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        Uri fileUri = FileProvider.getUriForFile(
                this,
                getString(R.string.file_provider_authorities),
                file);

        intent.setDataAndType(fileUri, "image/*");

        List<ResolveInfo> resInfoList = getPackageManager().queryIntentActivities(
                intent,
                PackageManager.MATCH_DEFAULT_ONLY);
        for (ResolveInfo info : resInfoList) {
            grantUriPermission(
                    info.activityInfo.packageName,
                    fileUri, FLAG_GRANT_WRITE_URI_PERMISSION | FLAG_GRANT_READ_URI_PERMISSION);
        }

        NotificationCompat.Builder notificationBuilder;
        NotificationManager notificationManager = (NotificationManager) this
                .getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (notificationManager != null) {
                notificationManager.createNotificationChannel(createChannel());
            }
            notificationBuilder = new NotificationCompat.Builder(this, CHANNEL_ID);
        } else {
            notificationBuilder = new NotificationCompat.Builder(this);
        }

        notificationBuilder
                .setContentTitle(getString(R.string.app_name))
                .setContentText(getString(R.string.notification_image_saved_click_to_preview))
                .setTicker(getString(R.string.notification_image_saved))
                .setSmallIcon(R.drawable.ic_done)
                .setOngoing(false)
                .setContentIntent(PendingIntent.getActivity(this, 0, intent, 0))
                .setAutoCancel(true);
        if (notificationManager != null) {
            notificationManager.notify(DOWNLOAD_NOTIFICATION_ID_DONE, notificationBuilder.build());
        }

        // WE TRIED PENG.
            //inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "tf_model.pb");
            // initialize File object
    //        File temp_file = new File("C:\\Users\\user\\Desktop\\AEyeAllianceV6(with model)\\uCrop-master\\sample\\src\\main\\assets\\tf_model.pb");

            // get the file path
    //        File fPath = file.getAbsoluteFile();

    //        inferenceInterface = new TensorFlowInferenceInterface(new FileInputStream(fPath));
    //        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);


        // Load the model
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "tf_model.pb");



    }


            @TargetApi(Build.VERSION_CODES.O)
    public NotificationChannel createChannel() {
        int importance = NotificationManager.IMPORTANCE_LOW;
        NotificationChannel channel = new NotificationChannel(CHANNEL_ID, getString(R.string.channel_name), importance);
        channel.setDescription(getString(R.string.channel_description));
        channel.enableLights(true);
        channel.setLightColor(Color.YELLOW);
        return channel;
    }

}
