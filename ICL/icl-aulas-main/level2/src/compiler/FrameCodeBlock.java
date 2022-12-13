package compiler;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a frame.
 */
public class FrameCodeBlock extends CodeBlock{
    
    private static final String CLASS_NAME = "f%d";

    private static final String START =

    ".class f%d\n" +
    ".super java/lang/Object\n" +
    ".field public sl %s\n";

    private static final String FRAME_DEPTH_0_SL = "Ljava/lang/Object;";

    private static final String FRAME_SL = "Lf%d;";

    private static final String END =

    ".method public <init>()V\n" +

    "  aload_0\n" +
    "  invokenonvirtual java/lang/Object/<init>()V\n" +
    "  return\n\n" +
    ".end method";

    public static final String FIELD_NAME_FORMAT = "x%d";
    private static final String FIELD_FORMAT = ".field public " + FIELD_NAME_FORMAT + " %s\n";


    public static final int INVALID_FRAME_ID = -1;

    private int frameId;

    private List<String> fieldsTypes;

    public final String slType;

    public final FrameCodeBlock previous;
    public FrameCodeBlock(int frameId, int numFields, FrameCodeBlock previous)
    {
        
        super(String.format(CLASS_NAME, frameId));
        this.previous = previous;
        this.frameId = frameId;
        this.fieldsTypes = new ArrayList<>(numFields);

        if (previous == null)
            slType = FRAME_DEPTH_0_SL;
        else
            slType = String.format(FRAME_SL, previous.getFrameId());
    }

    /**
     * @return the frameId
     */
    public int getFrameId() {
        return frameId;
    } 

    /**
     * Add a field to this frame
     * @param jvmType
     */
    public void addFieldType(String jvmType)
    {
        this.fieldsTypes.add(jvmType);
    }

    @Override
    public void dump(PrintStream f) { // dumps code to f

        f.printf(START, this.frameId, slType);

        for (int i = 0; i < this.fieldsTypes.size(); i++)
            f.printf(FIELD_FORMAT, i, this.fieldsTypes.get(i));

        f.print(END);

        f.flush();
    }
}
