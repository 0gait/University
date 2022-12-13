package compiler;

import java.io.PrintStream;

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

    private static final String VARIABLE = ".field public x%d I\n";

    public static final String VARIABLE_NAME = "x%d";

    public static final int INVALID_PREVIOUS_FRAME_ID = -1;

    private int frameId, previousFrameId;

    private int numVariables;

    private String className;

    public FrameCodeBlock(int frameId, int previousFrameId)
    {
        this.frameId = frameId;
        this.previousFrameId = previousFrameId;
        this.className = String.format(CLASS_NAME, frameId);
        this.numVariables = 0;
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

    /**
     * @return the numVariables
     */
    public int getNumVariables() {
        return numVariables;
    }

    /**
     * @param numVariables the numVariables to set
     */
    public void setNumVariables(int numVariables) {
        this.numVariables = numVariables;
    }

    public void dump(PrintStream f) { // dumps code to f

        if (this.previousFrameId == INVALID_PREVIOUS_FRAME_ID)
            f.printf(START, this.frameId, FRAME_DEPTH_0_SL);
        else
            f.printf(START, this.frameId, String.format(FRAME_SL, this.previousFrameId));

        for (int i = 0; i < this.numVariables; i++)
            f.printf(VARIABLE, i);

        f.print(END);

        f.flush();
    }
}
