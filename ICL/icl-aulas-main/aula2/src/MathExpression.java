import compiler.*;

import java.io.*;
import java.util.List;
import java.util.LinkedList;

/**
 * Main class for the Math Expression parser (interpreter & compiler).
 */
public class MathExpression {

	private static final String INTERPRETER_PARAM = "-e";
	private static final String COMPILER_PARAM = "-c";

	private static final String DEST_FOLDER_PARAM = "-d";
	private static final String JASMIN_JAR_PATH_ENV = "JASMIN_JAR_PATH";
	private static final String EXPRESSION_FILE_EXTENSION = ".icl";

	/** Main entry point. */
	public static void main(String args[]) {
		if (args.length == 0) {
			printUsage();
			System.exit(1);
		}

		String[] newArgs = new String[args.length - 1];
		System.arraycopy(args, 1, newArgs, 0, newArgs.length);

		if (args[0].equals(INTERPRETER_PARAM))
			interpreterMain(newArgs);
		else if (args[0].equals(COMPILER_PARAM))
			compilerMain(newArgs);
		else {
			System.err.println("Invalid argument.");
			printUsage();
			System.exit(1);
		}
	}

	/**
	 * Print usage.
	 */
	private static void printUsage() {
		String usage = "Usage:\n" + "\tjava -jar MathExpression.jar -c <input-file-name>.icl [OPTIONS]\n"
				+ "\tjava -jar MathExpression.jar -e\n\n" +

				"\t-c -> Compile an expression from a file to generate a .class file that computes that expression.\n"
				+ "\t-e -> Evaluate an expression from stdin.\n\n" +

				"OPTIONS:\n" + "\t" + DEST_FOLDER_PARAM + " <Output folder> ";

		System.out.println(usage);
	}

	/**
	 * The main for the interpreter.
	 * @param args
	 */
	private static void interpreterMain(String[] args) {
		if (args.length > 0) {
			printUsage();
			System.exit(1);
		}

		Parser parser = new Parser(System.in);
		boolean end = false;

		while (!end) {
			try {
				System.out.print("> ");
				ASTNode ast = parser.Start();
				System.out.println(ast.eval());
			} catch (ParseException e) {
				System.err.println("Syntax Error!");
				parser.ReInit(System.in);
			} catch (Exception e) {
				System.err.println("An error occurred!");
				System.err.println(e.getMessage());
				end = true;
			}
		}
	}

	/**
	 * The main for the compiler.
	 * @param args
	 */
	private static void compilerMain(String[] args) {
		if (!(args.length == 1 || args.length == 3)) {
			printUsage();
			System.exit(1);
		}

		String source = args[0];
		String destFolder = null;

		if (args.length == 3) {
			if (!args[1].equals(DEST_FOLDER_PARAM)) {
				System.err.println("Invalid argument.");
				printUsage();
				System.exit(1);
			}

			destFolder = args[2];
		}

		try {
			String jasminJarPath = getJasminPath();
			// Open expression file
			File expressionFile = new File(source);
			String expressionFileName = getExpressionFileNameWithoutExtension(expressionFile);

			// Init parser from the input file.
			Parser parser = new Parser(new FileInputStream(expressionFile));

			// Create destFolder
			if (destFolder != null)
				new File(destFolder).mkdirs();

			// Create tmp file to save CompileBlock output.
			File tmpFile = File.createTempFile(expressionFileName, ".j");
			tmpFile.deleteOnExit();

			ASTNode ast = parser.Start();

			// Compile expression and dump to tmp file.
			CompileBlock c = new CompileBlock(expressionFileName);
			ast.compile(c);
			c.dump(new PrintStream(tmpFile));

			// Call jasmin to compile to a .class file.

			List<String> jasminCommand = new LinkedList<String>();
			jasminCommand.add("java");
			jasminCommand.add("-jar");
			jasminCommand.add(jasminJarPath);

			if (destFolder != null) {
				jasminCommand.add(DEST_FOLDER_PARAM);
				jasminCommand.add(destFolder);
			}

			jasminCommand.add(tmpFile.getAbsolutePath());

			ProcessBuilder processBuilder = new ProcessBuilder(jasminCommand);
			Process jasminProcess = processBuilder.start();
			int jasminExitValue = jasminProcess.waitFor();

			if (jasminExitValue == 0) {
				System.out.println("Success!");
				System.out.println("Created file: " + expressionFileName + ".class");
			} else {
				System.err.println("Jasmin returned an error.");
				System.exit(jasminExitValue);
			}
		} catch (ParseException e) {
			System.err.println("Syntax Error!");
		} catch (Exception e) {
			System.err.println("An error occurred!");
			System.err.println(e.getMessage());
		}
	}

	/**
	 * Get the jasmin.jar file path from the environment.
	 * @return The path for the jasmin.jar file.
	 */
	private static String getJasminPath() {
		String jasminJarPath = System.getenv(JASMIN_JAR_PATH_ENV);
		if (jasminJarPath == null) {
			System.err.println("Cannot find jasmin jar executable.\n" + "Environment var " + JASMIN_JAR_PATH_ENV
					+ " must be defined.");
			System.exit(1);
		}

		return jasminJarPath;
	}

	/**
	 * Get the name of the file without the extension.
	 * @param expressionFile - The file to get the name.
	 * @return The name of the file without the extension.
	 */
	private static String getExpressionFileNameWithoutExtension(File expressionFile) {
		String name = expressionFile.getName();
		int extensionIndex;
		if (name.isEmpty() || (extensionIndex = name.lastIndexOf(EXPRESSION_FILE_EXTENSION)) <= 0)
			throw new IllegalArgumentException("The input file name must begin with a valid character"
					+ " and have the " + EXPRESSION_FILE_EXTENSION + " extension.");

		return name.substring(0, extensionIndex);
	}
}