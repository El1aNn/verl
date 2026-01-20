def solve_akari(rows, cols, grid_str):
    """
    解决 Akari 谜题。

    Args:
        rows: 谜题的行数。
        cols: 谜题的列数。
        grid_str: 代表谜题的字符串。

    Returns:
        解决后的谜题字符串。
    """

    grid = [list(grid_str[i * cols:(i + 1) * cols]) for i in range(rows)]
    empty_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '.':
                empty_cells.append((r, c))

    def is_valid(r, c):
        """检查在 (r, c) 放置灯泡是否有效。"""
        # 检查左边
        for i in range(c - 1, -1, -1):
            if grid[r][i] in '01234X':
                break
            if grid[r][i] == 'L':
                return False
        # 检查右边
        for i in range(c + 1, cols):
            if grid[r][i] in '01234X':
                break
            if grid[r][i] == 'L':
                return False
        # 检查上面
        for i in range(r - 1, -1, -1):
            if grid[i][c] in '01234X':
                break
            if grid[i][c] == 'L':
                return False
        # 检查下面
        for i in range(r + 1, rows):
            if grid[i][c] in '01234X':
                break
            if grid[i][c] == 'L':
                return False
        return True

    def check_solution():
        """检查当前灯泡布局是否是有效的解。"""
        # 检查所有数字方块的条件
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] in '01234':
                    required_lamps = int(grid[r][c])
                    adjacent_lamps = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 'L':
                            adjacent_lamps += 1
                    if adjacent_lamps != required_lamps:
                        return False

        # 检查所有白格是否被照亮
        illuminated = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'L':
                    illuminated.add((r, c))
                    # 向左照亮
                    for i in range(c - 1, -1, -1):
                        if grid[r][i] in '01234X':
                            break
                        illuminated.add((r, i))
                    # 向右照亮
                    for i in range(c + 1, cols):
                        if grid[r][i] in '01234X':
                            break
                        illuminated.add((r, i))
                    # 向上照亮
                    for i in range(r - 1, -1, -1):
                        if grid[i][c] in '01234X':
                            break
                        illuminated.add((i, c))
                    # 向下照亮
                    for i in range(r + 1, rows):
                        if grid[i][c] in '01234X':
                            break
                        illuminated.add((i, c))

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '.' and (r, c) not in illuminated:
                    return False
        return True


    def solve(index):
        """递归回溯求解。"""
        if index == len(empty_cells):
            if check_solution():
                return "".join(["".join(row) for row in grid])
            return None

        r, c = empty_cells[index]

        # 尝试在 (r, c) 放置灯泡
        if is_valid(r, c):
            grid[r][c] = 'L'
            solution = solve(index + 1)
            if solution:
                return solution
            grid[r][c] = '.'  # 回溯

        # 尝试不在 (r, c) 放置灯泡
        solution = solve(index + 1)
        if solution:
            return solution

        return None

    return solve(0)

if __name__ == '__main__':
    try:
        rows, cols = map(int, input().split())
        grid_str = input()
        solution = solve_akari(rows, cols, grid_str)
        if solution:
            print(solution)
    except (ValueError, IndexError):
        print("无效的输入格式。")