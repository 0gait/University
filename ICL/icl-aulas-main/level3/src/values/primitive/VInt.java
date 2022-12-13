package values.primitive;

import types.primitives.TypeInt;
import types.primitives.TypePrimitive;

public final class VInt extends VPrimitive<Integer>
{
    public final static String TYPE_NAME = TypeInt.TYPE.show();

    /**
     * @param value
     */
    public VInt(int value) {
        super(value);
    }

    @Override
    public TypePrimitive getPrimitiveType() {
        return (TypePrimitive)TypeInt.TYPE;
    }
}
