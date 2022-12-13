package values;

/**
 * Represents a memory cell.
 */
public class VCell implements IValue
{
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
}
