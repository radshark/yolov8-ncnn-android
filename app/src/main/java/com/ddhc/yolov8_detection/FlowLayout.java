package com.ddhc.yolov8_detection;

import android.content.Context;
import android.content.DialogInterface;
import android.content.res.TypedArray;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.LayerDrawable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.GestureDetector;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

/**
 * Created by zm
 */
public class FlowLayout extends ViewGroup {

    String TAG = "ncnn";
    private Context mContext;
    private int usefulWidth; // the space of a line we can use(line's width minus the sum of left and right padding
    private int lineSpacing = 0; // the spacing between lines in flowlayout
    List<View> childList = new ArrayList();
    List<Integer> lineNumList = new ArrayList();
    private OnChildClickListener childClickListener;

    public FlowLayout(Context context) {
        this(context, null);
    }

    public FlowLayout(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    HashMap<String, Integer> labelMap = new HashMap<>();
    Path path;
    MarginLayoutParams marginLayoutParams = new MarginLayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT);
    GestureDetector gestureDetector;

    public FlowLayout(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        Log.i(TAG, "FlowLayout: " + 6);


        mContext = context;

        marginLayoutParams.setMarginEnd(10);
        Log.i(TAG, "FlowLayout: 加载存储类型");
        try {
            //加载存储类型
            //cls argb
            path = Paths.get(context.getFilesDir() + "/cls.txt");
//            Files.deleteIfExists(path);
            if (Files.exists(path)) {
                List<String> strings = Files.readAllLines(path);
                for (String string : strings) {
                    String[] split = string.split(" ");
                    if (split.length == 2) { // Ensure we have enough parts
                        String label = split[0];
                        int argb = Integer.parseInt(split[1]);
                        labelMap.put(label, argb);
                        TextView textView = new TextView(this.getContext());
                        textView.setText(label);
                        textView.setPadding(16, 10, 16, 10);
                        textView.setBackgroundColor(argb);
                        this.addView(textView, marginLayoutParams);
                    }
                }
            } else {
                Files.createFile(path);
            }
        } catch (IOException e) {
            // Log the error instead of throwing
            Log.e(TAG, "Error handling the file", e);
            try {
                Files.deleteIfExists(path);
            } catch (IOException ex) {
                Log.e("FileError", "Error deleting file", ex);
            }
        }
        //添加新增类型按钮
        TextView textView = new TextView(this.getContext());
        textView.setPadding(16, 10, 16, 10);
        textView.setWidth(120);
        textView.setText("+");
        textView.setGravity(Gravity.CENTER);
        // 创建一个带有边框的Drawable
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(Color.TRANSPARENT); // 设置背景颜色为透明
        drawable.setStroke(4, Color.RED, 10, 4); // 设置边框的宽度和颜色

// 创建一个LayerDrawable，并将边框Drawable作为层添加
        LayerDrawable layerDrawable = new LayerDrawable(new Drawable[]{drawable});

// 设置TextView的背景为LayerDrawable
        textView.setBackground(layerDrawable);

        addView(textView);

        TypedArray mTypedArray = context.obtainStyledAttributes(attrs, R.styleable.FlowLayout);
        lineSpacing = mTypedArray.getDimensionPixelSize(R.styleable.FlowLayout_lineSpacing, 0);
        mTypedArray.recycle();

        //控制器
        gestureDetector = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onSingleTapUp(MotionEvent event) {
                // 检查哪个子控件被点击
                for (int i = 0; i < getChildCount(); i++) {
                    TextView child = (TextView) getChildAt(i);
                    if (isPointInsideView2(event.getX(), event.getY(), child)) {
                        //增加类型被点击
                        if (i == getChildCount() - 1) {
                            showInputDialog(context);
                        }
                        // cls子控件被点击
                        else {
                            String cls = child.getText().toString();
                            childClickListener.onChildClick(cls, labelMap.get(cls));
                        }
                    }
                }
                return true;
            }

            @Override
            public void onLongPress(MotionEvent event) {
                Log.d("ncnn", "onLongPress");
                // 检查哪个子控件被长按
                for (int i = 0; i < getChildCount(); i++) {
                    TextView child = (TextView) getChildAt(i);
                    if (isPointInsideView2(event.getX(), event.getY(), child)) {
                        Log.d("ncnn", "Child clicked: " + i);
                        //增加类型被点击
                        if (i == getChildCount() - 1) {
                            //showInputDialog(context);
                        }
                        // cls子控件被点击
                        else {
                            String label = child.getText().toString();
                            longClick(child, label, context);
                        }
                    }
                }
            }
        });
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
//        Log.i(TAG, "onTouchEvent: GestureDetector");
        // 将触摸事件传递给 GestureDetector
        gestureDetector.onTouchEvent(event);
//        gestureDetector.onTouchEvent(event);
        return true;
    }

    private void longClick(TextView child, String label, Context context) {
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setPositiveButton("确认", (dialog, which) -> {

            try {
                //清空存储
                StringBuffer buffer = new StringBuffer();
                labelMap.forEach((k, va) -> {
                    if (!k.equals(label)) {
                        buffer.append(k).append(" ").append(va).append("\n");
                    }
                });
                Files.write(path, Collections.emptyList());
                //重新写入
                Files.write(path, buffer.toString().getBytes(StandardCharsets.UTF_8));
                //移除view及map
                removeView(child);
                labelMap.remove(label);
            } catch (IOException e) {
                Log.i(TAG, "删除失败");
            }
        }).setTitle("删除 " + label + " ?").setCancelable(true).setNegativeButton("取消", null).show();
    }

    private void showInputDialog(Context context) {
        /*@setView 装入一个EditView
         */
        final EditText editText = new EditText(context);
        AlertDialog.Builder inputDialog = new AlertDialog.Builder(context);
        Random random = new Random();
        inputDialog.setTitle("添加分类").setView(editText);
        inputDialog.setPositiveButton("确定", (dialog, which) -> {
            String label = editText.getText().toString();
            if (labelMap.containsKey(label)) {
                Toast.makeText(context, "类型已存在...", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!label.equals("") && label.length() < 13 && label.length() >= 2) {
                try {
                    int argb = Color.argb(150, random.nextInt(255), random.nextInt(255), random.nextInt(255));
                    Files.write(path, (label + " " + argb + "\n").getBytes(StandardCharsets.UTF_8), StandardOpenOption.APPEND);
                    TextView textView = new TextView(context);
                    textView.setText(label);
                    labelMap.put(label, argb);
                    textView.setPadding(16, 10, 16, 10);
                    textView.setBackgroundColor(argb);
                    addView(textView, getChildCount() - 1, marginLayoutParams);
                } catch (IOException e) {
                    Toast.makeText(context, "新增类型失败...", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(context, "请输入名称长度2~12字符", Toast.LENGTH_SHORT).show();
            }
        }).setCancelable(true).show();
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int mPaddingLeft = getPaddingLeft();
        int mPaddingRight = getPaddingRight();
        int mPaddingTop = getPaddingTop();
        int mPaddingBottom = getPaddingBottom();

        int widthSize = MeasureSpec.getSize(widthMeasureSpec);
        int heightMode = MeasureSpec.getMode(heightMeasureSpec);
        int heightSize = MeasureSpec.getSize(heightMeasureSpec);
        int lineUsed = mPaddingLeft + mPaddingRight;
        int lineY = mPaddingTop;
        int lineHeight = 0;
        for (int i = 0; i < this.getChildCount(); i++) {
            View child = this.getChildAt(i);
            if (child.getVisibility() == GONE) {
                continue;
            }
            int spaceWidth = 0;
            int spaceHeight = 0;
            LayoutParams childLp = child.getLayoutParams();
            if (childLp instanceof MarginLayoutParams) {
                measureChildWithMargins(child, widthMeasureSpec, 0, heightMeasureSpec, lineY);
                MarginLayoutParams mlp = (MarginLayoutParams) childLp;
                spaceWidth = mlp.leftMargin + mlp.rightMargin;
                spaceHeight = mlp.topMargin + mlp.bottomMargin;
            } else {
                measureChild(child, widthMeasureSpec, heightMeasureSpec);
            }

            int childWidth = child.getMeasuredWidth();
            int childHeight = child.getMeasuredHeight();
            spaceWidth += childWidth;
            spaceHeight += childHeight;

            if (lineUsed + spaceWidth > widthSize) {
                //approach the limit of width and move to next line
                lineY += lineHeight + lineSpacing;
                lineUsed = mPaddingLeft + mPaddingRight;
                lineHeight = 0;
            }
            if (spaceHeight > lineHeight) {
                lineHeight = spaceHeight;
            }
            lineUsed += spaceWidth;
        }
        setMeasuredDimension(widthSize, heightMode == MeasureSpec.EXACTLY ? heightSize : lineY + lineHeight + mPaddingBottom);
    }


    private boolean isPointInsideView(float x, float y, View view) {
        Log.i(TAG, "isPointInsideView1: ");
        int[] location = new int[2];
        view.getLocationOnScreen(location);
        int left = location[0];
        int top = location[1];
        return x >= left && x <= left + view.getWidth() && y >= top && y <= top + view.getHeight();
    }

    private boolean isPointInsideView2(float x, float y, View view) {
        Log.i(TAG, "isPointInsideView2: ");
        return x >= view.getX() && x <= view.getX() + view.getWidth() && y >= view.getY() && y <= view.getY() + view.getHeight();
    }

    @Override
    protected void onLayout(boolean changed, int l, int t, int r, int b) {
        int mPaddingLeft = getPaddingLeft();
        int mPaddingRight = getPaddingRight();
        int mPaddingTop = getPaddingTop();

        int lineX = mPaddingLeft;
        int lineY = mPaddingTop;
        int lineWidth = r - l;
        usefulWidth = lineWidth - mPaddingLeft - mPaddingRight;
        int lineUsed = mPaddingLeft + mPaddingRight;
        int lineHeight = 0;
        int lineNum = 0;
        lineNumList.clear();
        for (int i = 0; i < this.getChildCount(); i++) {
            View child = this.getChildAt(i);
            if (child.getVisibility() == GONE) {
                continue;
            }
            int spaceWidth = 0;
            int spaceHeight = 0;
            int left = 0;
            int top = 0;
            int right = 0;
            int bottom = 0;
            int childWidth = child.getMeasuredWidth();
            int childHeight = child.getMeasuredHeight();

            LayoutParams childLp = child.getLayoutParams();
            if (childLp instanceof MarginLayoutParams) {
                MarginLayoutParams mlp = (MarginLayoutParams) childLp;
                spaceWidth = mlp.leftMargin + mlp.rightMargin;
                spaceHeight = mlp.topMargin + mlp.bottomMargin;
                left = lineX + mlp.leftMargin;
                top = lineY + mlp.topMargin;
                right = lineX + mlp.leftMargin + childWidth;
                bottom = lineY + mlp.topMargin + childHeight;
            } else {
                left = lineX;
                top = lineY;
                right = lineX + childWidth;
                bottom = lineY + childHeight;
            }
            spaceWidth += childWidth;
            spaceHeight += childHeight;

            if (lineUsed + spaceWidth > lineWidth) {
                //approach the limit of width and move to next line
                lineNumList.add(lineNum);
                lineY += lineHeight + lineSpacing;
                lineUsed = mPaddingLeft + mPaddingRight;
                lineX = mPaddingLeft;
                lineHeight = 0;
                lineNum = 0;
                if (childLp instanceof MarginLayoutParams) {
                    MarginLayoutParams mlp = (MarginLayoutParams) childLp;
                    left = lineX + mlp.leftMargin;
                    top = lineY + mlp.topMargin;
                    right = lineX + mlp.leftMargin + childWidth;
                    bottom = lineY + mlp.topMargin + childHeight;
                } else {
                    left = lineX;
                    top = lineY;
                    right = lineX + childWidth;
                    bottom = lineY + childHeight;
                }
            }
            child.layout(left, top, right, bottom);
            lineNum++;
            if (spaceHeight > lineHeight) {
                lineHeight = spaceHeight;
            }
            lineUsed += spaceWidth;
            lineX += spaceWidth;
        }
        // add the num of last line
        lineNumList.add(lineNum);
    }

    /**
     * resort child elements to use lines as few as possible
     */
    public void relayoutToCompress() {
        post(new Runnable() {
            @Override
            public void run() {
                compress();
            }
        });
    }

    private void compress() {
        int childCount = this.getChildCount();
        if (0 == childCount) {
            //no need to sort if flowlayout has no child view
            return;
        }
        int count = 0;
        for (int i = 0; i < childCount; i++) {
            View v = getChildAt(i);
            if (v instanceof BlankView) {
                //BlankView is just to make childs look in alignment, we should ignore them when we relayout
                continue;
            }
            count++;
        }
        View[] childs = new View[count];
        int[] spaces = new int[count];
        int n = 0;
        for (int i = 0; i < childCount; i++) {
            View v = getChildAt(i);
            if (v instanceof BlankView) {
                //BlankView is just to make childs look in alignment, we should ignore them when we relayout
                continue;
            }
            childs[n] = v;
            LayoutParams childLp = v.getLayoutParams();
            int childWidth = v.getMeasuredWidth();
            if (childLp instanceof MarginLayoutParams) {
                MarginLayoutParams mlp = (MarginLayoutParams) childLp;
                spaces[n] = mlp.leftMargin + childWidth + mlp.rightMargin;
            } else {
                spaces[n] = childWidth;
            }
            n++;
        }
        int[] compressSpaces = new int[count];
        for (int i = 0; i < count; i++) {
            compressSpaces[i] = spaces[i] > usefulWidth ? usefulWidth : spaces[i];
        }
        sortToCompress(childs, compressSpaces);
        this.removeAllViews();
        for (View v : childList) {
            this.addView(v);
        }
        childList.clear();
    }

    private void sortToCompress(View[] childs, int[] spaces) {
        int childCount = childs.length;
        int[][] table = new int[childCount + 1][usefulWidth + 1];
        for (int i = 0; i < childCount + 1; i++) {
            for (int j = 0; j < usefulWidth; j++) {
                table[i][j] = 0;
            }
        }
        boolean[] flag = new boolean[childCount];
        for (int i = 0; i < childCount; i++) {
            flag[i] = false;
        }
        for (int i = 1; i <= childCount; i++) {
            for (int j = spaces[i - 1]; j <= usefulWidth; j++) {
                table[i][j] = (table[i - 1][j] > table[i - 1][j - spaces[i - 1]] + spaces[i - 1]) ? table[i - 1][j] : table[i - 1][j - spaces[i - 1]] + spaces[i - 1];
            }
        }
        int v = usefulWidth;
        for (int i = childCount; i > 0 && v >= spaces[i - 1]; i--) {
            if (table[i][v] == table[i - 1][v - spaces[i - 1]] + spaces[i - 1]) {
                flag[i - 1] = true;
                v = v - spaces[i - 1];
            }
        }
        int rest = childCount;
        View[] restArray;
        int[] restSpaces;
        for (int i = 0; i < flag.length; i++) {
            if (flag[i] == true) {
                childList.add(childs[i]);
                rest--;
            }
        }

        if (0 == rest) {
            return;
        }
        restArray = new View[rest];
        restSpaces = new int[rest];
        int index = 0;
        for (int i = 0; i < flag.length; i++) {
            if (flag[i] == false) {
                restArray[index] = childs[i];
                restSpaces[index] = spaces[i];
                index++;
            }
        }
        table = null;
        childs = null;
        flag = null;
        sortToCompress(restArray, restSpaces);
    }

    /**
     * add some blank view to make child elements look in alignment
     */
    public void relayoutToAlign() {
        post(new Runnable() {
            @Override
            public void run() {
                align();
            }
        });
    }

    private void align() {
        int childCount = this.getChildCount();
        if (0 == childCount) {
            //no need to sort if flowlayout has no child view
            return;
        }
        int count = 0;
        for (int i = 0; i < childCount; i++) {
            View v = getChildAt(i);
            if (v instanceof BlankView) {
                //BlankView is just to make childs look in alignment, we should ignore them when we relayout
                continue;
            }
            count++;
        }
        View[] childs = new View[count];
        int[] spaces = new int[count];
        int n = 0;
        for (int i = 0; i < childCount; i++) {
            View v = getChildAt(i);
            if (v instanceof BlankView) {
                //BlankView is just to make childs look in alignment, we should ignore them when we relayout
                continue;
            }
            childs[n] = v;
            LayoutParams childLp = v.getLayoutParams();
            int childWidth = v.getMeasuredWidth();
            if (childLp instanceof MarginLayoutParams) {
                MarginLayoutParams mlp = (MarginLayoutParams) childLp;
                spaces[n] = mlp.leftMargin + childWidth + mlp.rightMargin;
            } else {
                spaces[n] = childWidth;
            }
            n++;
        }
        int lineTotal = 0;
        int start = 0;
        this.removeAllViews();
        for (int i = 0; i < count; i++) {
            if (lineTotal + spaces[i] > usefulWidth) {
                int blankWidth = usefulWidth - lineTotal;
                int end = i - 1;
                int blankCount = end - start;
                if (blankCount >= 0) {
                    if (blankCount > 0) {
                        int eachBlankWidth = blankWidth / blankCount;
                        MarginLayoutParams lp = new MarginLayoutParams(eachBlankWidth, 0);
                        for (int j = start; j < end; j++) {
                            this.addView(childs[j]);
                            BlankView blank = new BlankView(mContext);
                            this.addView(blank, lp);
                        }
                    }
                    this.addView(childs[end]);
                    start = i;
                    i--;
                    lineTotal = 0;
                } else {
                    this.addView(childs[i]);
                    start = i + 1;
                    lineTotal = 0;
                }
            } else {
                lineTotal += spaces[i];
            }
        }
        for (int i = start; i < count; i++) {
            this.addView(childs[i]);
        }
    }

    /**
     * use both of relayout methods together
     */
    public void relayoutToCompressAndAlign() {
        post(new Runnable() {
            @Override
            public void run() {
                compress();
                align();
            }
        });
    }

    /**
     * cut the flowlayout to the specified num of lines
     *
     * @param line_num_now
     */
    public void specifyLines(final int line_num_now) {
        post(new Runnable() {
            @Override
            public void run() {
                int line_num = line_num_now;
                int childNum = 0;
                if (line_num > lineNumList.size()) {
                    line_num = lineNumList.size();
                }
                for (int i = 0; i < line_num; i++) {
                    childNum += lineNumList.get(i);
                }
                List<View> viewList = new ArrayList<>();
                for (int i = 0; i < childNum; i++) {
                    viewList.add(getChildAt(i));
                }
                removeAllViews();
                for (View v : viewList) {
                    addView(v);
                }
            }
        });
    }

    @Override
    protected LayoutParams generateLayoutParams(LayoutParams p) {
        return new MarginLayoutParams(p);
    }

    @Override
    public LayoutParams generateLayoutParams(AttributeSet attrs) {
        return new MarginLayoutParams(getContext(), attrs);
    }

    @Override
    protected LayoutParams generateDefaultLayoutParams() {
        return new MarginLayoutParams(super.generateDefaultLayoutParams());
    }

    public void setOnChildClickListener(OnChildClickListener onChildClickListener) {
        this.childClickListener = onChildClickListener;
    }

    class BlankView extends View {

        public BlankView(Context context) {
            super(context);
        }
    }

    public static interface OnChildClickListener {

        public void onChildClick(String cls, int color);
    }
}