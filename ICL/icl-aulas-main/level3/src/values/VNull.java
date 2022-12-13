package values;

import types.IType;
import types.TypeNull;

/**
 * A special value in the language that represents null
 * This value cannot be used in any operation and throws a Runtime Exception.
 */
public class VNull implements IValue
{
    public final static String TYPE_NAME = TypeNull.TYPE.show();

    private VNull() {}

    public static final VNull VALUE = new VNull();

    @Override
    public String show() {
        return TypeNull.TYPE.show();
    }

    @Override
    public IType getType() {
        return TypeNull.TYPE;
    }
    @Override
    public String toString() {
       return show();
    }

    
}
