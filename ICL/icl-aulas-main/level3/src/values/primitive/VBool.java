package values.primitive;

import types.primitives.TypeBool;
import types.primitives.TypePrimitive;

public final class VBool extends VPrimitive<Boolean>
{
    public final static String TYPE_NAME = TypeBool.TYPE.show();

    /**
     * @param value
     */
    public VBool(boolean value) {
        super(value);
    }

    @Override
    public TypePrimitive getPrimitiveType() {
        return (TypePrimitive)TypeBool.TYPE;
    }
    
}
