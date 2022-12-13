package ast.record;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import compiler.RecordCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeRecord;

import util.Bind;
import util.Pair;
import util.Utils;
import values.IValue;
import values.VRecord;

public class ASTRecord extends ASTNodeAbstract
{
    private List<Pair<Bind,IType>> fields;

    /**
     * @param fields
     */
    public ASTRecord(List<Pair<Bind,IType>> fields) {
        this.fields = fields;
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        Map<String, IValue> recordMap = new LinkedHashMap<>(fields.size());

        for (Pair<Bind,IType> field : fields) {
            Bind bind = field.getLeft();
            recordMap.put(bind.getLeft(), Utils.requireNonNull(bind.getRight().eval(e)));
        }

        if(recordMap.size() != fields.size())
            throw new TypeErrorException("Cannot have repeated names for different fields in the same record.")
                .toRuntimeException();
        
        return new VRecord(recordMap);
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e) {
       RecordCodeBlock record =  c.createRecord((TypeRecord)this.type);
       for (Pair<Bind,IType> pair : fields) {
           Bind bind = pair.getLeft();
           bind.getRight().compile(c, e);
           c.emit(String.format("putfield %s/%s %s", record.className,bind.getLeft(), bind.getRight().getType().getJvmType()));
           c.emit("dup");
       }
       c.emit("pop");

    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException
    {
        Map<String, IType> recordMap = new LinkedHashMap<>(fields.size());

        for (Pair<Bind,IType> field : fields)
        {
            Bind bind = field.getLeft();
            IType declaredType = field.getRight();
            IType actualType = Utils.requireNonNull(bind.getRight().typecheck(e));

            // Check if declared type equals the actual type
            if (declaredType != null && !declaredType.equals(actualType))
                throw new TypeErrorException(
                    String.format("Illegal expression type in record def for bind with id '%s'. " +
                        "Declared type is '%s' and got '%s'.", bind.getLeft(), declaredType.show(),
                        actualType.show()));
                
            recordMap.put(bind.getLeft(), actualType);
        }

        if(recordMap.size() != fields.size())
            throw new TypeErrorException("Cannot have repeated names for different fields in the same record.");
		
        return this.type =  new TypeRecord(recordMap);
    }

    @Override
    public StringBuilder toString(StringBuilder builder)
    {
		Utils.toStringList(this.fields, 
			(field) -> {
				builder.append(field.getLeft().getLeft());
				builder.append('=');
				((ASTNodeAbstract)field.getLeft().getRight()).toString(builder);
			},
			null, new String[]{"[", "]"}, builder);

		return builder;
    }
}
