package ast.names;

import java.util.List;

import ast.ASTNode;
import ast.ASTNodeAbstract;
import compiler.FrameCodeBlock;
import compiler.MainCodeBlock;
import types.IType;
import util.Bind;
import util.Coordinates;
import util.Environment;
import util.Pair;
import values.IValue;

public class ASTDef extends ASTNodeAbstract {
	private List<Bind> init;
	private ASTNode body;

	public ASTDef(List<Bind> init, ASTNode body) {
		this.init = init;
		this.body = body;
	}

	@Override
	public IValue eval(Environment<IValue> e)
	{
		Environment<IValue> env = e.beginScope();

		for (Bind bind : this.init)
		{
			env.assoc(bind.getLeft(), bind.getRight().eval(env));
		}

		IValue value = body.eval(env);
		env.endScope();
		return value;
	}

	@Override
	public IType typecheck(Environment<IType> e)
	{
		Environment<IType> env = e.beginScope();

		for (Bind bind : this.init)
		{
			env.assoc(bind.getLeft(), bind.getRight().typecheck(env));
		}

		IType bodyType = body.typecheck(env);
		type = bodyType;
		env.endScope();
		return bodyType;
	}

	@Override
	public void compile(MainCodeBlock c, Environment<Coordinates> e)
	{
		Pair<Environment<Coordinates>, FrameCodeBlock> pairEnvFrameId = c.addFrame(e, this.init.size());
		Environment<Coordinates> newEnv = pairEnvFrameId.getLeft();
		FrameCodeBlock frame = pairEnvFrameId.getRight();

		IType type;
		String jvmType;
		int i = 0;
		for (Bind bind : this.init)
		{
			type = bind.getRight().getType();
			jvmType = type.getJvmType();
			frame.addFieldType(jvmType);

			Coordinates varCoord = new Coordinates(frame.getFrameId(), String.format(FrameCodeBlock.FIELD_NAME_FORMAT, i));

			c.emitCurrentFrame();
			bind.getRight().compile(c, newEnv);
			c.emit(String.format("putfield f%d/%s %s", frame.getFrameId(), varCoord.getRight(), jvmType));

			newEnv.assoc(bind.getLeft(), varCoord);
			i++;
		}

		this.body.compile(c, newEnv);
		c.endFrame(newEnv);
	}
}
