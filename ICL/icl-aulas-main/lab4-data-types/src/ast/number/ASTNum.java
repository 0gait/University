package ast.number;

import ast.ASTNodeAbstract;
import compiler.MainCodeBlock;
import types.IType;
import types.primitves.TypeInt;
import util.Coordinates;
import util.Environment;
import values.IValue;
import values.primitive.VInt;

public class ASTNum extends ASTNodeAbstract {

	private int val;

	public ASTNum(int n) {
		val = n;
		type = TypeInt.TYPE;
	}

	public IValue eval(Environment<IValue> e) {
		return new VInt(this.val);
	}
	
	@Override
	public void compile(MainCodeBlock c, Environment<Coordinates> e) {
		c.emit("sipush " + val);
	}

	@Override
	public IType typecheck(Environment<IType> e) {
		return type;
	}

	

}
