package values.primitive;

import types.primitives.TypePrimitive;
import types.primitives.TypeString;

public final class VString extends VPrimitive<String>
{
    public final static String TYPE_NAME = TypeString.TYPE.show();

    public VString(String value) {
        super(value);
    }

    @Override
    public TypePrimitive getPrimitiveType() {
        return (TypePrimitive)TypeString.TYPE;
    }
    
}
