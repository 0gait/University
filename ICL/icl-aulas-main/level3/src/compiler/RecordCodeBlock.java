package compiler;

import java.io.PrintStream;
import java.util.Map;


import types.IType;
import types.TypeRecord;

public class RecordCodeBlock extends CodeBlock
{
    private static final String CLASS_NAME = "record%d";

    private static final String START =

    ".class " +  CLASS_NAME +"\n" +
    ".super java/lang/Object\n";


    private static final String END =

    ".method public <init>()V\n" +

    "  aload_0\n" +
    "  invokenonvirtual java/lang/Object/<init>()V\n" +
    "  return\n\n" +
    ".end method";

    private static final String FIELD_FORMAT = ".field public %s %s\n";


    private int recordId;
    private TypeRecord type;

    public RecordCodeBlock(int recordId, TypeRecord type)
    {
        super(String.format(CLASS_NAME, recordId));
        this.recordId = recordId;
        this.type = type;
    }
    /**
     * @return the frameId
     */
    public int getRecordId() {
        return recordId;
    } 


    @Override
    public void dump(PrintStream f) { // dumps code to f

        f.printf(START, this.recordId);

        for ( Map.Entry<String,IType> entry : type) {
            f.printf(FIELD_FORMAT, entry.getKey(), entry.getValue().getJvmType());
            
        }
        f.print(END);

        f.flush();
    }
}
