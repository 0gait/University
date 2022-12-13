package ast.record;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.Coordinates;
import compiler.MainCodeBlock;
import compiler.RecordCodeBlock;
import environment.Environment;
import typeError.TypeErrorException;
import types.IType;
import types.TypeRecord;
import util.Utils;
import values.IValue;
import values.VRecord;

public class ASTGetRecordValue extends ASTNodeAbstract
{
    private static final String OPERATOR = ".";

    private final ASTNode record;
    private final String id;

    /**
     * @param record
     * @param id
     */
    public ASTGetRecordValue(ASTNode record, String id) {
        this.record = record;
        this.id = id;
    }

    @Override
    public IValue eval(Environment<IValue> e)
    {
        VRecord record = Utils.checkValueForOperation(this.record.eval(e), VRecord.class, OPERATOR);
        return record.getValue(this.id);
       
    }

    @Override
    public void compile(MainCodeBlock c, Environment<Coordinates> e)
    {
        record.compile(c, e);

        RecordCodeBlock record = c.getRecord((TypeRecord) this.record.getType());
        c.emit("checkcast " + record.className);
        c.emit(String.format("getfield %s/%s %s", record.className, id, this.type.getJvmType()));
    }

    @Override
    public IType typecheck(Environment<IType> e) throws TypeErrorException {
        TypeRecord type = Utils.checkTypeForOperation(this.record.typecheck(e), TypeRecord.class, OPERATOR);
         this.type = type.getFieldTypeFromRecord(id);
         if(this.type == null)
            throw new TypeErrorException("No field in record with id = " + id);
        return this.type;
    }

    @Override
    public StringBuilder toString(StringBuilder builder)
    {
        ((ASTNodeAbstract)this.record).toString(builder);
        builder.append('.');
        builder.append(this.id);

        return builder;
    }    
}
