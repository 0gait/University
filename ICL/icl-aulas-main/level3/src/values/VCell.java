package values;

import types.IType;
import types.TypeRef;

/**
 * Represents a memory cell.
 */
public class VCell implements IValue
{
    public final static String TYPE_NAME = TypeRef.TYPE_NAME;

    protected IValue value;

    /**
     * @param value
     */
    public VCell(IValue value) {
        this.value = value;
    }

    @Override
    public String show() {
        return "Ref ( " + this.value.show() + " ) ";
    }

    /**
     * @return the value
     */
    public IValue getValue() {
        return value;
    }

    /**
     * @param value the value to set
     */
    public void setValue(IValue value) {
        this.value = value;
    }

    @Override
    public IType getType() {
        return new TypeRef(this.value.getType());
    }

    @Override
    public String toString() {
       return show();
    }
}
