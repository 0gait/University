package values;

import java.util.Map;
import java.util.NoSuchElementException;

import types.IType;
import types.TypeRecord;
import util.Utils;

public class VRecord implements IValue
{
    public final static String TYPE_NAME = TypeRecord.TYPE_NAME;

    private Map<String, IValue> record;

    public VRecord(Map<String, IValue> record)
    {
        this.record = record;
    }

    public IValue getValue(String id)
    {
        IValue value = this.record.get(id);
        if (value == null)
            throw new NoSuchElementException("No field in record with id = " + id);

        return value;
    }

    @Override
    public String show() {
        StringBuilder builder = new StringBuilder(TYPE_NAME + " ");

        return Utils.toStringList(record.entrySet(),
        (entry) -> {
            builder.append(entry.getKey());
            builder.append('=');
            builder.append(entry.getValue().show());
        }, null, new String[]{"[", "]"}, builder).toString();
    }

    @Override
    public String toString() {
       return show();
    }

    @Override
    public IType getType() {
        return TYPE;
    }

    public static final IType TYPE = new IType() {

        @Override
        public String show() {
            return this.toString();
        }

        @Override
        public String getJvmType() {
            return "";
        }

        @Override
        public String toString() {

            return "Record";
        }

        @Override
        public String getSimpleName() {
            // TODO Auto-generated method stub
            return null;
        }
    };
    
}
