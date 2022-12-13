package compiler;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

public class FrameCodeBlock {
    
    private static final String CLASS_NAME = "f%d";

    private static final String START =

    ".class f%d\n" +
    ".super java/lang/Object\n" +
    ".field public sl %s;\n";

    private static final String FRAME_DEPTH_0_SL = "Ljava/lang/Object";

    private static final String FRAME_SL = "Lf%d";

    private static final String END =

    ".method public <init>()V\n" +

    "  aload_0\n" +
    "  invokenonvirtual java/lang/Object/<init>()V\n" +
    "  return\n\n" +
    ".end method";

    public static final String FIELD_NAME_FORMAT = "x%d";
    private static final String FIELD_FORMAT = ".field public " + FIELD_NAME_FORMAT + " %s\n";


    public static final int INVALID_PREVIOUS_FRAME_ID = -1;

    private int frameId, previousFrameId;

    private String className;

    private List<String> fieldsTypes;

    public FrameCodeBlock(int frameId, int previousFrameId, int numFields)
    {
        this.frameId = frameId;
        this.previousFrameId = previousFrameId;
        this.className = String.format(CLASS_NAME, frameId);
        this.fieldsTypes = new ArrayList<>(numFields);
    }

    /**
     * @return the frameId
     */
    public int getFrameId() {
        return frameId;
    } 

    /**
     * @return the className
     */
    public String getClassName() {
        return className;
    }

    public void addFieldType(String jvmType)
    {
        this.fieldsTypes.add(jvmType);
    }

    public void dump(PrintStream f) { // dumps code to f

        if (this.previousFrameId == INVALID_PREVIOUS_FRAME_ID)
            f.printf(START, this.frameId, FRAME_DEPTH_0_SL);
        else
            f.printf(START, this.frameId, String.format(FRAME_SL, this.previousFrameId));

        for (int i = 0; i < this.fieldsTypes.size(); i++)
            f.printf(FIELD_FORMAT, i, this.fieldsTypes.get(i));

        f.print(END);

        f.flush();
    }
}
