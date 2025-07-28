module.exports = async ({ github, context }) => {
  const fs = require("fs");

  let data = fs.readFileSync("docstring-adder-diff.txt", { encoding: "utf8" });
  // posting comment fails if too long, so truncate
  if (data.length > 30000) {
    let truncated_data = data.substring(0, 30000);
    let lines_truncated =
      data.split("\n").length - truncated_data.split("\n").length;
    data =
      truncated_data + `\n\n... (truncated ${lines_truncated} lines) ...\n`;
  }

  const body = data.trim()
    ? "⚠ Diff showing the effect of this PR on the typeshed stdlib: \n```diff\n" +
      data +
      "\n```"
    : "This change has no effect on the docstrings that would be added to typeshed. 🤖🎉";
  const issue_number = parseInt(
    fs.readFileSync("pr_number.txt", { encoding: "utf8" }),
  );
  await github.rest.issues.createComment({
    issue_number,
    owner: context.repo.owner,
    repo: context.repo.repo,
    body,
  });

  return issue_number;
};
