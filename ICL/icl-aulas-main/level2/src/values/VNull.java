package values;

public class VNull implements IValue {
    
    private VNull() {}

    public static final VNull VALUE = new VNull();

    @Override
    public String show() {
        return "Null";
    }

}
