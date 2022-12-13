package types;


import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import util.Utils;


public class TypeRecord  implements IType, Iterable<Entry<String,IType>> {

    public static final String TYPE_NAME = "Record";
    protected Map<String,IType> fields;
    public TypeRecord(Map<String,IType> fields){
        this.fields = Collections.unmodifiableMap(fields);
    }
    @Override
    public String show() {
    
        StringBuilder builder = new StringBuilder(TYPE_NAME + " ");

        return Utils.toStringList(fields.entrySet(),
        (entry) -> {
            builder.append(entry.getKey());
            builder.append(':');
            builder.append(entry.getValue().show());
        }, null, new String[]{"[", "]"}, builder).toString();
       
    }

    @Override
    public String getJvmType() {
      
        return "Ljava/lang/Object;";
    }

    @Override

    public String toString(){
        return  show();
    }

    @Override
    public String getSimpleName() {
        return  TYPE_NAME;
    }
    

    @Override
    public int hashCode() {
        return show().hashCode();
    }
    
    @Override
    public boolean equals(Object obj) {
        if( this == obj)
            return true;
        if(!(obj instanceof TypeRecord))
            return false;
        TypeRecord other = (TypeRecord)obj;
       return this.fields.equals(other.fields);
    }

    public IType getFieldTypeFromRecord(String id){
        return fields.get(id);
    }
    @Override
    public Iterator<Entry<String, IType>> iterator() {
        return fields.entrySet().iterator();
    }
   
}